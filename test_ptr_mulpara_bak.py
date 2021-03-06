import os, time, pickle, copy,sys

import numpy as np
import tensorflow as tf
from random import randint

from collections import defaultdict
from tensorflow.python.ops import array_ops
from tensorflow.contrib.legacy_seq2seq import sequence_loss

from ops import l2_loss, weight
from utils import progress
from ptr_cell_para import PTRCell, mod, D, batch_size


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

flags = tf.app.flags

flags.DEFINE_integer("N", 10, "memory size")
flags.DEFINE_integer("W", 10, "memory dimensions")
flags.DEFINE_integer("K", 1, "key dimensions")
flags.DEFINE_integer("G", 1, "number of registers")

tf.flags.DEFINE_float("epsilon", 1e-08, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate")

flags.DEFINE_integer("epochs", 200000, "Epoch to train [100000]")
flags.DEFINE_integer("input_dim", 10, "Dimension of input [10]")
flags.DEFINE_integer("output_dim", 10, "Dimension of output [10]")
flags.DEFINE_integer("min_length", 1, "Minimum length of input sequence [1]")
flags.DEFINE_integer("max_length", 6, "Maximum length of output sequence [10]")
flags.DEFINE_integer("controller_layer_size", 2, "The size of controller [1]")
flags.DEFINE_integer("test_max_length", 6, "Maximum length of output sequence [10]")
flags.DEFINE_string("checkpoint_dir", "/data/leisha/checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_integer("print_interval", 50, "print info interval")

FLAGS = flags.FLAGS


class PTRModel(object):
    def __init__(self, I, O, N, W, K, G, sess, min_length = 1, max_length = 10,
                 controller_layer_size=1, min_grad=-10, max_grad=+10,
                 lr=1e-3, epsilon=1e-8, momentum=0.9, decay=0.0, weight_decay=0, scope="PTRModel", forward_only=False):
        self.controller = PTRCell([(batch_size, N,W), (batch_size,N,W), (batch_size,N*N,W)], [1, 1, 0], [0, 0, 1], (1, W), controller_layer_size=controller_layer_size, addr_mode=0)
        self.sess = sess
        self.scope = scope

        self.input_dim = I
        self.output_dim = O
        self.W = W
        
        self.min_length = min_length
        self.max_length = max_length
        self.min_grad = min_grad
        self.max_grad = max_grad
        self.weight_decay = weight_decay

        self.MAX_STEP = max_length*max_length+1

        self.inputs_1 = []
        self.inputs_2 = []
        self.true_outputs = []

        self.outputs = {}
        self.prev_states = {}

        self.losses = {}
        self.optims = {}
        self.grads = {}

        self.collect_states = {0:[]}
        self.debug = {0:[]}

        self.saver = None
        self.params = None

        self.global_step = tf.Variable(0, trainable=False)
        #self.opt = tf.train.RMSPropOptimizer(lr, decay=decay, epsilon=epsilon, momentum=momentum)
        #self.opt = tf.train.AdagradOptimizer(lr)
        self.opt = tf.train.AdamOptimizer(lr)
        #self.opt = tf.train.AdamOptimizer(lr, epsilon=epsilon)
        self.build_model(forward_only)

    def build_model(self, forward_only):
        print("[*] Building a PTRModel math model")

        with tf.variable_scope(self.scope):
            self.a = weight('a', [1, 1])
            # self.c = weight('c', [1, 1])
            # self.d = weight('d', [1, 1])
            self.b = weight('b', [1, 1], init='constant')
            self.beta = 1 + tf.nn.softplus(weight('beta', [1, 1]))




            for seq_length in range(1, self.max_length*self.max_length + 1):
                true_output = tf.placeholder(tf.float32, [batch_size, self.output_dim],
                                             name='true_output_%s' % seq_length)
                self.true_outputs.append(true_output)

            prev_state = self.controller.init_state(self.true_outputs[0].get_shape()[0])

            tf.get_variable_scope().reuse_variables()

            for seq_length in range(1, self.max_length + 1):
                input_1 = tf.placeholder(tf.float32, [batch_size, self.input_dim],
                                         name='input_1_%s' % seq_length)
                input_2 = tf.placeholder(tf.float32, [batch_size, self.input_dim],
                                         name='input_2_%s' % seq_length)

                self.inputs_1.append(input_1)
                self.inputs_2.append(input_2)

                # present inputs
                prev_state = self.controller.update_memory(prev_state, [tf.reshape(input_1, [batch_size, 1, -1]), tf.reshape(input_2, [batch_size, 1, -1]),  tf.zeros((batch_size, 1, self.W))])
                self.collect_states[seq_length] = self.collect_states[seq_length-1][0:(seq_length-1)] + [self.copy_state(prev_state)]

                self.debug[seq_length] = []


                state = prev_state
                self.prev_states[seq_length] = state

                candidate_outputs = []
                for j in range(0, self.MAX_STEP):
                    state = self.controller(state,j)
                    new_state = self.copy_state(state)
                    self.collect_states[seq_length].append(new_state)
                    # print(state['M'][-1][:,0:(seq_length*seq_length)].get_shape())
                    candidate_outputs.append(tf.unstack(tf.transpose(state['M'][-1][:,0:(seq_length*seq_length)], [1,0,2])))
                    # self.debug[seq_length].append((new_state['ptr'],new_state['dptr']))


                self.outputs[seq_length] = candidate_outputs

            if not forward_only:
                for seq_length in range(self.min_length, self.max_length + 1):
                    print(" [*] Building a loss model for seq_length %s" % seq_length)
                    print(len(self.outputs[seq_length]),len(self.true_outputs[0:seq_length*seq_length]),len([1] * (seq_length*seq_length)))
                    # print(self.outputs[seq_length][0].shape,self.true_outputs[0:2*seq_length][0].shape,len([1] * (2*seq_length)))
                    all_losses = []
                    for index in range(self.MAX_STEP):
                        # print(len(self.outputs[seq_length][index]), self.outputs[seq_length][index][0].get_shape())
                        # print(len(tf.unstack(self.true_outputs[0:seq_length*seq_length])), tf.unstack(self.true_outputs[0:seq_length*seq_length])[0].get_shape())

                        loss = sequence_loss(logits=self.outputs[seq_length][index],
                                             targets=tf.unstack(self.true_outputs[0:seq_length*seq_length]),
                                             weights=[[1]*batch_size]* seq_length*seq_length,
                                             average_across_timesteps=False,
                                             average_across_batch=False,
                                             softmax_loss_function=l2_loss)
                        all_losses.append(loss)

                    all_losses = tf.stack(all_losses)

                    cn = tf.pow(tf.to_float(seq_length), self.a) + self.b
                    max_pos = tf.clip_by_value(cn, 0, self.MAX_STEP-1)
                    stop_pos = D(self.MAX_STEP, tf.tile(max_pos, [batch_size,1]), 1, self.beta)

                    loss1 = tf.reduce_sum(tf.expand_dims(all_losses, 0) * stop_pos) + 0.0001 * tf.reduce_sum(cn)

                    self.losses[seq_length] = loss1

                    if not self.params:
                        self.params = tf.trainable_variables()

                    grads = []
                    for grad in tf.gradients(loss1, self.params): # + self.weight_decay*tf.add_n(tf.get_collection('l2'))
                        if grad is not None:
                            grads.append(tf.clip_by_value(grad,
                                                          self.min_grad,
                                                          self.max_grad))
                        else:
                            grads.append(grad)
                    self.grads[seq_length] = grads

        with tf.variable_scope("opt", reuse=None):
            if not forward_only:
                for seq_length in range(self.min_length, self.max_length + 1):
                    self.optims[seq_length] = self.opt.apply_gradients(
                                                           zip(self.grads[seq_length], self.params),
                                                           global_step=self.global_step)

        self.saver = tf.train.Saver()
        print(" [*] Build a PTRModel math model finished")

    def get_outputs(self, seq_length, index):
        return self.outputs[seq_length][index]


    def get_stop_pos(self, seq_length):
        #stop_pos = tf.nn.softmax(tf.nn.embedding_lookup(self.length2stepdist, [seq_length]) * 50)
        max_pos = tf.clip_by_value(tf.pow(tf.to_float(seq_length), self.a)  + self.b, 0, self.MAX_STEP - 1)
        stop_pos = D(self.MAX_STEP, tf.tile(max_pos, [batch_size,1]), 1, self.beta)
        return tf.argmax(stop_pos[0][0]), max_pos, self.a, self.b, self.beta

    def get_loss(self, seq_length):
        if seq_length not in self.losses:
            index, _,   _, _ = self.get_stop_pos(seq_length)
            # loss = sequence_loss(logits=self.outputs[seq_length][index],
            #                      targets=self.true_outputs[0:seq_length],
            #                      weights=[1] * seq_length,
            #                      average_across_timesteps=False,
            #                      average_across_batch=False,
            #                      softmax_loss_function=l2_loss)

            loss = sequence_loss(logits=self.outputs[seq_length][index],
                                 targets=tf.unstack(self.true_outputs[0:seq_length * seq_length]),
                                 weights=[[1] * batch_size] * seq_length * seq_length,
                                 average_across_timesteps=False,
                                 average_across_batch=False,
                                 softmax_loss_function=l2_loss)
            self.losses[seq_length] = loss / batch_size / seq_length
        return self.losses[seq_length]

    def copy_state(self, state):
        new_state = {}
        for k,v in state.items():
            if k != 'seq_length':
                new_state[k] = v
        return new_state
 
    def get_collect_state(self, seq_length):
        return self.collect_states[seq_length]

    def save(self, checkpoint_dir, task_name, step):
        task_dir = os.path.join(checkpoint_dir, "%s" % (task_name))
        file_name = "PTRModel_%s.model" % task_name

        if not os.path.exists(task_dir):
            os.makedirs(task_dir)

        self.saver.save(self.sess,
                       os.path.join(task_dir, file_name),
                       global_step = step.astype(int))

    def load(self, checkpoint_dir, task_name):
        print(" [*] Reading checkpoints...")

        checkpoint_dir = os.path.join(checkpoint_dir, "%s" % (task_name))

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        else:
            raise Exception(" [!] Testing, but %s not found" % checkpoint_dir)

def generate_sequence(length, bits):
    seq = np.zeros([length, bits], dtype=np.float32)
    for idx in range(length):
        seq[idx, 0:bits] = np.round(np.random.rand(bits) * 3)
    return list(seq)

def op(l1, l2):
    return mul(l1, l2)

def elem_add(l1, l2):
    res = []
    for i,j in zip(l1, l2):
        res.append(i+j)
    return res

def add(l1, l2, base=10):
    res = []
    a = np.zeros([1])
    for i, j in zip(l1, l2):
        b = (i + j + a) % base
        a = (i + j + a) // base
        res.append(b)
    # print('l1:',l1,'l2:',l2,"res:" ,res)
    return res

def list2int(l):
    n = 0
    for i in l[::-1]:
        a = i[0]
        n = n * 10 + a

    return n

def int2list(n, seq_len):
    n=int(n)
    s = str(n)
    # print(n,s)
    li = list(s)[::-1]
    res = [np.array([int(a)]) for a in li]
    while len(res) < seq_len * 2:
        res.append(np.array([0]))
    return res


def mul(l1, l2, base=10):
    resnum = []
    for n1 in l1:
        for n2 in l2:
            resnum.append(n1*n2)
    return resnum

def elem_mul(l1, l2):
    res = []
    for i, j in zip(l1, l2):
        res.append(i*j)
    return res

def elem_mod(l1, l2, base=10):
    res = []
    for i,j in zip(l1, l2):
        res.append((i+j) % base)
    return res

def onehot(l, length, batch):
    if batch == 1:
        res = np.zeros((length, 10))
        for i in range(length):
                res[i,  np.int(l[i][0])] = 1.0
        return list(res)
    else:
        res = np.zeros((length,batch, 10))
        for i in range(length):
            for b in range(batch):
                res[i,b, np.int(l[i][b])] = 1.0
        return list(res)

def pprint(seq):
    seq = np.array(seq)
    dim = (seq.shape[1] - 1)//2
    seq = np.char.mod('%d', np.around(seq))
    seq[:, dim:dim+1] = ' '
    print("\n".join(["".join(x) for x in seq.tolist()]))

def test(model, seq_length, sess, print_=True):
    seq1 = generate_sequence(seq_length, batch_size)
    seq2 = generate_sequence(seq_length, batch_size)
    seq = op(seq1, seq2)

    feed_dict = {input_:vec for vec, input_ in zip(onehot(seq1, seq_length,batch_size), model.inputs_1)}
    feed_dict.update(
        {input_:vec for vec, input_ in zip(onehot(seq2, seq_length,batch_size), model.inputs_2)}
    )
    feed_dict.update(
        {true_output:vec for vec, true_output in zip(onehot(seq, seq_length*seq_length,batch_size), model.true_outputs)}
    )
    #feed_dict.update({true_output:vec for vec, true_output in zip(seq, model.true_outputs)})

    #, model.get_collect_state(seq_length)
    stop_pos, max_pos, a,   b, beta = sess.run(model.get_stop_pos(seq_length), feed_dict=feed_dict)

    print('test pos:', stop_pos, max_pos, a,   b, beta)
    outputs, loss, states = sess.run([model.get_outputs(seq_length, stop_pos), model.get_loss(seq_length), model.get_collect_state(seq_length)], feed_dict=feed_dict)

    outputs = list(np.reshape(np.argmax(np.asarray(outputs)[:,0,:], 1), [seq_length*seq_length, 1]))
    f = open('test_case_outp_%d' % seq_length, 'wb')
    pickle.dump([seq1, seq2, seq, outputs, states], f)
    f.close()

    debug = sess.run(model.debug[seq_length], feed_dict=feed_dict)

    if print_:
        np.set_printoptions(suppress=True)
        print(" true and predicted outputs : ")
        # print(np.asarray(seq)[:,0].shape)
        # print(np.zeros((seq_length*seq_length, 1)).shape)
        # print( np.asarray(outputs).shape)
        pprint(np.hstack([np.asarray(seq)[:,:1], np.zeros((seq_length*seq_length, 1)), np.asarray(outputs)]))
        print(" Loss : %f" % loss)
        np.set_printoptions(suppress=False)
        # for ptr in debug:
        #     # print( ptr)
        #     print (mod(ptr[0], seq_length))
    else:
        return seq, outputs, loss

def train(sess):
    if not os.path.isdir(FLAGS.checkpoint_dir):
        raise Exception(" [!] Directory %s not found" % FLAGS.checkpoint_dir)

    # delimiter flag for start and end
    model = PTRModel(FLAGS.input_dim, FLAGS.output_dim, FLAGS.N, FLAGS.W, FLAGS.K, FLAGS.G,
              sess, min_length=FLAGS.min_length, max_length=FLAGS.max_length,
              lr=FLAGS.learning_rate, epsilon=FLAGS.epsilon, controller_layer_size=FLAGS.controller_layer_size)

    print(" [*] Initialize all variables")
    model.load(FLAGS.checkpoint_dir, 'outp' + sys.argv[1] + '_' + sys.argv[2])
    #sess.run(tf.global_variables_initializer())
    print(" [*] Initialization finished")

    batch_size_num = batch_size

    start_time = time.time()
    for idx in range(FLAGS.epochs):
        #seq_length = randint(FLAGS.min_length, FLAGS.max_length)
        lens = sys.argv[1]
        lengths = [int(a) for a in lens.split(',')]
        for num in lengths:
            seq_length = num
            seq1 = generate_sequence(seq_length, batch_size_num)
            # print(seq1,onehot(seq1, seq_length))
            seq2 = generate_sequence(seq_length, batch_size_num)
            seq = op(seq1, seq2)

            feed_dict = {input_:vec for vec, input_ in zip(onehot(seq1, seq_length,batch_size_num), model.inputs_1)}
            feed_dict.update(
                {input_:vec for vec, input_ in zip(onehot(seq2, seq_length, batch_size_num), model.inputs_2)}
            )
            feed_dict.update(
                {true_output:vec for vec, true_output in zip(onehot(seq, seq_length*seq_length, batch_size_num), model.true_outputs)}
            )
            #feed_dict.update({true_output:vec for vec, true_output in zip(seq, model.true_outputs)})

            _, cost, step = sess.run([model.optims[seq_length],
                                      model.get_loss(seq_length),
                                      model.global_step], feed_dict=feed_dict)

        if idx % 100 == 0 or idx == FLAGS.epochs-1:
            model.save(FLAGS.checkpoint_dir, 'outp' + sys.argv[1] + '_' + sys.argv[2], step)

        seq_length = int(sys.argv[2])
        if idx % FLAGS.print_interval == 0:
            print("[%5d] seq_length(%2d): %.2f (%.1fs)" \
                % (idx, seq_length, cost, time.time() - start_time))

            test(model, seq_length, sess)

    print("Training math task finished")
    return model

def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if FLAGS.is_train:
            model = train(sess)
        else:
            model = PTRModel(FLAGS.input_dim, FLAGS.output_dim, FLAGS.N, FLAGS.W, FLAGS.K, FLAGS.G,
                      sess, min_length=FLAGS.min_length, max_length=FLAGS.max_length,
                      lr=FLAGS.learning_rate, epsilon=FLAGS.epsilon, controller_layer_size=FLAGS.controller_layer_size)
            model.load(FLAGS.checkpoint_dir, 'outp' + sys.argv[1] + '_' + sys.argv[2])

            test(model, int(FLAGS.test_max_length/3), sess)
            test(model, int(FLAGS.test_max_length*2/3), sess)
            test(model, int(FLAGS.test_max_length*3/3), sess)

if __name__ == '__main__':
    tf.app.run()

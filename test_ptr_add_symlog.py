import os, time, pickle, copy,sys

import numpy as np
import tensorflow as tf
from random import randint

from collections import defaultdict
from tensorflow.python.ops import array_ops
from tensorflow.contrib.legacy_seq2seq import sequence_loss

from ops import l2_loss, weight
from utils import progress
from ptr_cell import PTRCell, D

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
flags = tf.app.flags

flags.DEFINE_integer("N", 12, "memory size")
flags.DEFINE_integer("W", 10, "memory dimensions")
flags.DEFINE_integer("K", 1, "key dimensions")
flags.DEFINE_integer("G", 1, "number of registers")

tf.flags.DEFINE_float("epsilon", 1e-08, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")

flags.DEFINE_integer("epochs", 200000, "Epoch to train [100000]")
flags.DEFINE_integer("input_dim", 10, "Dimension of input [10]")
flags.DEFINE_integer("output_dim", 10, "Dimension of output [10]")
flags.DEFINE_integer("min_length", 1, "Minimum length of input sequence [1]")
flags.DEFINE_integer("max_length", 12, "Maximum length of output sequence [10]")
flags.DEFINE_integer("controller_layer_size", 2, "The size of controller [1]")
flags.DEFINE_integer("test_max_length", 12, "Maximum length of output sequence [10]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_integer("print_interval", 10, "print info interval")
flags.DEFINE_integer("MAX_STEP", 12, "max steps to train")

FLAGS = flags.FLAGS

class PTRModel(object):
    def __init__(self, I, O, N, W, K, G, sess, min_length = 1, max_length = 12,
                 controller_layer_size=1, min_grad=-10, max_grad=+10,
                 lr=1e-3, epsilon=1e-8, momentum=0.9, decay=0.0, weight_decay=0, MAX_STEP = 12, scope="PTRModel", forward_only=False):
        self.controller = PTRCell([(N,W), (N,W), (N,W)], [1, 1, 0], [0, 0, 1], (1, W), controller_layer_size=controller_layer_size, addr_mode=0)
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

        self.MAX_STEP = MAX_STEP

        self.inputs_1 = []
        self.inputs_2 = []
        self.true_outputs = []

        self.stops = {}
        self.outputs = {}
        self.prev_states = {}

        self.losses = {}
        self.optims = {}
        self.grads = {}

        self.collect_states = {0:[]}

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
            #self.length2stepdist = weight('embedding', [self.max_length+1, self.MAX_STEP])   # for each length we have a vector with max_length representing the step dist
            self.a = weight('exp', [1,1])
            self.b = weight('exp_weight', [1, 1])
            self.c = weight('exp_bias', [1,1], init = 'constant')
            self.beta = 1 + tf.nn.softplus(weight('beta', [1,1]))

            prev_state = self.controller.init_state()

            tf.get_variable_scope().reuse_variables()
            for seq_length in range(1, self.max_length + 1):
                input_1 = tf.placeholder(tf.float32, [self.input_dim],
                                         name='input_1_%s' % seq_length)
                input_2 = tf.placeholder(tf.float32, [self.input_dim],
                                         name='input_2_%s' % seq_length)
                true_output = tf.placeholder(tf.float32, [self.output_dim],
                                             name='true_output_%s' % seq_length)

                self.inputs_1.append(input_1)
                self.inputs_2.append(input_2)
                self.true_outputs.append(true_output)

                # present inputs
                prev_state = self.controller.update_memory(prev_state, [tf.reshape(input_1, [1, -1]), tf.reshape(input_2, [1, -1]), tf.zeros((1, self.W))])
                self.collect_states[seq_length] = self.collect_states[seq_length-1][0:(seq_length-1)] + [self.copy_state(prev_state)]

                state = prev_state
                self.prev_states[seq_length] = state

                stops = []
                candidate_outputs = []
                for j in range(self.MAX_STEP):
                    state, stop = self.controller(state,j)
                    self.collect_states[seq_length].append(self.copy_state(state))
                    candidate_outputs.append(tf.unstack(state['M'][-1][0:seq_length]))
                    # stops.append(state['stop'])

                self.outputs[seq_length] = candidate_outputs
                self.stops[seq_length] = stops

            if not forward_only:
                for seq_length in range(self.min_length, self.max_length + 1):
                    print(" [*] Building a loss model for seq_length %s" % seq_length)
                    all_losses = []
                    for index in range(self.MAX_STEP):
                        loss = sequence_loss(logits=self.outputs[seq_length][index],
                                             targets=self.true_outputs[0:seq_length],
                                             weights=[1] * seq_length,
                                             average_across_timesteps=False,
                                             average_across_batch=False,
                                             softmax_loss_function=l2_loss)
                        all_losses.append(loss)

                    all_losses = tf.stack(all_losses)

                    #step_dist = tf.nn.softmax(tf.concat(self.stops[seq_length], 1))
                    #step_dist = tf.nn.softmax(tf.nn.embedding_lookup(self.length2stepdist, [seq_length]))
                    #max_pos = tf.to_float(tf.argmax(step_dist))
                    cn = tf.pow(tf.to_float(seq_length), self.a) * tf.pow(tf.log(tf.to_float(seq_length)), self.b) + self.c
                    max_pos = tf.clip_by_value(cn, 0, self.MAX_STEP-1)
                    stop_pos = D(self.MAX_STEP, max_pos, 1, self.beta)

                    loss1 = tf.reduce_sum(tf.expand_dims(all_losses, 0) * stop_pos) + 0.001 * tf.reduce_sum(cn)

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
        max_pos = tf.clip_by_value(tf.pow(tf.to_float(seq_length), self.a) * tf.pow(tf.log(tf.to_float(seq_length)), self.b) + self.c, 0, self.MAX_STEP - 1)
        stop_pos = D(self.MAX_STEP, max_pos, 1, self.beta)
        return tf.argmax(stop_pos[0]), max_pos, self.a, self.b, self.c, self.beta

    def get_loss(self, seq_length):
        if seq_length not in self.losses:
            index, _, _, _, _, _ = self.get_stop_pos(seq_length)
            loss = sequence_loss(logits=self.outputs[seq_length][index],
                                 targets=self.true_outputs[0:seq_length],
                                 weights=[1] * seq_length,
                                 average_across_timesteps=False,
                                 average_across_batch=False,
                                 softmax_loss_function=l2_loss)

            self.losses[seq_length] = loss
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
        seq[idx, 0:bits] = np.round(np.random.rand(bits) * 8)
    return list(seq)

def op(l1, l2):
    return add(l1, l2)

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
    return res

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

def onehot(l, length):
    res = np.zeros((length, 10))
    for i in range(length):
        res[i, np.int(l[i][0])] = 1.0
    return list(res)

def pprint(seq):
    seq = np.array(seq)
    dim = (seq.shape[1] - 1)//2
    seq = np.char.mod('%d', np.around(seq))
    seq[:, dim:dim+1] = ' '
    print("\n".join(["".join(x) for x in seq.tolist()]))

def test(model, seq_length, sess, print_=True):
    seq1 = generate_sequence(seq_length, 1)
    seq2 = generate_sequence(seq_length, 1)
    seq = op(seq1, seq2)

    feed_dict = {input_:vec for vec, input_ in zip(onehot(seq1, seq_length), model.inputs_1)}
    feed_dict.update({input_:vec for vec, input_ in zip(onehot(seq2, seq_length), model.inputs_2)})
    feed_dict.update({true_output:vec for vec, true_output in zip(onehot(seq, seq_length), model.true_outputs)})
    # feed_dict.update({model.input_seqlen: seq_length})
    #feed_dict.update({true_output:vec for vec, true_output in zip(seq, model.true_outputs)})

    #, model.get_collect_state(seq_length)
    stop_pos, max_pos, exp, exp_weight, exp_bias, beta = sess.run(model.get_stop_pos(seq_length), feed_dict=feed_dict)

    print('test pos:', stop_pos, max_pos, exp, exp_weight, exp_bias, beta)
    outputs, loss, states = sess.run([model.get_outputs(seq_length, stop_pos), model.get_loss(seq_length), model.get_collect_state(seq_length)], feed_dict=feed_dict)

    outputs = list(np.reshape(np.argmax(np.asarray(outputs), 1), [seq_length, 1]))
    f = open('test_case_%d' % seq_length, 'wb')
    pickle.dump([seq1, seq2, seq, outputs, states], f)
    f.close()

    # printTestScore(sess, model, seq_length)

    if print_:
        np.set_printoptions(suppress=True)
        print(" true and predicted outputs : ")
        pprint(np.hstack([np.asarray(seq), np.zeros((seq_length, 1)), np.asarray(outputs)]))
        print(" Loss : %f" % loss)
        np.set_printoptions(suppress=False)
    else:
        return seq, outputs, loss

def train(sess):
    if not os.path.isdir(FLAGS.checkpoint_dir):
        raise Exception(" [!] Directory %s not found" % FLAGS.checkpoint_dir)

    # delimiter flag for start and end
    model = PTRModel(FLAGS.input_dim, FLAGS.output_dim, FLAGS.N, FLAGS.W, FLAGS.K, FLAGS.G,
              sess, min_length=FLAGS.min_length, max_length=FLAGS.max_length, MAX_STEP = FLAGS.MAX_STEP,
              lr=FLAGS.learning_rate, epsilon=FLAGS.epsilon, controller_layer_size=FLAGS.controller_layer_size)

    print(" [*] Initialize all variables")

    # model.load(FLAGS.checkpoint_dir, 'mathlog' + sys.argv[1])
    sess.run(tf.global_variables_initializer())
    print(" [*] Initialization finished")

    start_time = time.time()
    for idx in range(FLAGS.epochs):
        #seq_length = randint(FLAGS.min_length, FLAGS.max_length)
        seq_length = int(sys.argv[1])#randint(3, 5)
        seq1 = generate_sequence(seq_length, 1)
        seq2 = generate_sequence(seq_length, 1)
        seq = op(seq1, seq2)
        # print(seq1)
        # print(onehot(seq1, seq_length))

        feed_dict = {input_:vec for vec, input_ in zip(onehot(seq1, seq_length), model.inputs_1)}
        feed_dict.update({input_:vec for vec, input_ in zip(onehot(seq2, seq_length), model.inputs_2)})
        feed_dict.update({true_output:vec for vec, true_output in zip(onehot(seq, seq_length), model.true_outputs)})
        # feed_dict.update({model.input_seqlen: seq_length})
        #feed_dict.update({true_output:vec for vec, true_output in zip(seq, model.true_outputs)})
        
        _, cost, step = sess.run([model.optims[seq_length],
                                  model.get_loss(seq_length),
                                  model.global_step], feed_dict=feed_dict)

        if idx % 100 == 0 or idx == FLAGS.epochs-1:
            model.save(FLAGS.checkpoint_dir, 'mathlog' + sys.argv[1], step)

        if idx % FLAGS.print_interval == 0:
            print("[%5d] seq_length(%2d): %.2f (%.1fs)" \
                % (idx, seq_length, cost, time.time() - start_time))

            test(model, seq_length, sess)
            # test(model, seq_length+1, sess)

    print("Training math task finished")
    return model

def Gen_test(seq_length):
    res = []
    for i in range(1000):
        seq1 = generate_sequence(seq_length, 1)
        seq2 = generate_sequence(seq_length, 1)
        seq = op(seq1, seq2)
        res.append([seq1, seq2, seq])

    return res
testset = Gen_test(int(sys.argv[1]))

def printTestScore(sess,model,seq_length):
    total = 0
    allright = 0
    digitright = 0
    digittotal = 0
    accu = 0.0
    for seq1, seq2, seq in testset:
        total += 1

        if total % 100 == 0:
            print(total,' done')
        digittotal += len(seq)
        feed_dict = {input_: vec for vec, input_ in zip(onehot(seq1, seq_length), model.inputs_1)}
        feed_dict.update({input_: vec for vec, input_ in zip(onehot(seq2, seq_length), model.inputs_2)})
        feed_dict.update({true_output: vec for vec, true_output in zip(onehot(seq, seq_length), model.true_outputs)})
        # feed_dict.update({model.input_seqlen: seq_length})
        # feed_dict.update({true_output:vec for vec, true_output in zip(seq, model.true_outputs)})

        # , model.get_collect_state(seq_length)
        stop_pos, max_pos, exp, exp_weight, exp_bias, beta = sess.run(model.get_stop_pos(seq_length), feed_dict=feed_dict)

        outputs, loss, states = sess.run(
            [model.get_outputs(seq_length, stop_pos), model.get_loss(seq_length), model.get_collect_state(seq_length)],
            feed_dict=feed_dict)
        outputs = list(np.reshape(np.argmax(np.asarray(outputs), 1), [seq_length, 1]))
        # print(seq,outputs)


        casedigitright = 0
        casedigittotal = len(seq)
        for a, o in zip(seq, outputs):
            if a[0] == o[0]:
                digitright += 1
                casedigitright += 1

        accu += 1.0*casedigitright / casedigittotal

        if casedigittotal == casedigitright:
            allright += 1

    print('EM:', 1.0*allright/total, 'DA:', 1.0 * digitright/digittotal, 'micro:', accu / total)


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        if FLAGS.is_train:
            model = train(sess)
        else:
            model = PTRModel(FLAGS.input_dim, FLAGS.output_dim, FLAGS.N, FLAGS.W, FLAGS.K, FLAGS.G,
                      sess, min_length=FLAGS.min_length, max_length=FLAGS.max_length, MAX_STEP=FLAGS.MAX_STEP,
                      lr=FLAGS.learning_rate, epsilon=FLAGS.epsilon, controller_layer_size=FLAGS.controller_layer_size)
            model.load(FLAGS.checkpoint_dir, 'mathlog')

            test(model, int(FLAGS.test_max_length/3), sess)
            test(model, int(FLAGS.test_max_length*2/3), sess)
            test(model, int(FLAGS.test_max_length*3/3), sess)

if __name__ == '__main__':
    tf.app.run()

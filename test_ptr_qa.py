import os, time, pickle, sys

import numpy as np
import tensorflow as tf
from random import randint
from tensorflow.python.ops import array_ops
from tensorflow.contrib.seq2seq import sequence_loss
from ops import l2_loss, weight
from utils import progress
from data_utils import load_task, vectorize_data
from six.moves import range
from functools import reduce
from sklearn import cross_validation, metrics
from itertools import chain

from ptr_cell import PTRCell

flags = tf.app.flags
flags.DEFINE_integer("N", 100, "memory size")
flags.DEFINE_integer("W", 15, "memory dimensions")
flags.DEFINE_integer("G", 1, "number of registers")

tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")

flags.DEFINE_integer("epochs", 100, "Epoch to train [100000]")
flags.DEFINE_integer("task_id", 3, "bAbI task id, 1 <= id <= 20")
flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en-10k/", "Directory containing bAbI tasks")
flags.DEFINE_integer("controller_layer_size", 2, "The size of controller [1]")
flags.DEFINE_integer("controller_hidden_size", 32, "The hidden size of LSTM controller in each layer [256]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_integer("batch_size", 10, "Batch size")
flags.DEFINE_integer("print_interval", 10, "print info interval")

FLAGS = flags.FLAGS


class PTRModel(object):
    def __init__(self, I, O, N, W, G, sess, batch_size=1024, story_size=1000, max_hops=3, 
                 controller_layer_size=1, controller_hidden_size=256,  min_grad=-10, max_grad=+10,
                 lr=1e-4, epsilon=0.1, weight_decay=0, scope="PTRModel", forward_only=False):
        self.controller = PTRCell([(N, W), (N, W)], [1, 1], [0, 0], (1, W), controller_layer_size=controller_layer_size, controller_hidden_size=controller_hidden_size, addr_mode=1)
     
        self.sess = sess
        self.scope = scope

        self.input_dim = I
        self.output_dim = O
        self.W = W

        self.batch_size = batch_size
        self.story_size = story_size
        self.max_hops = max_hops
        self.min_grad = min_grad
        self.max_grad = max_grad
        self.weight_decay = weight_decay

        self.outputs = []

        self.saver = None
        self.params = None

        self.global_step = tf.Variable(0, trainable=False)
        self.opt = tf.train.AdamOptimizer(lr, epsilon=epsilon)
        #self.opt = tf.train.RMSPropOptimizer(lr, decay=0.0, epsilon=epsilon, momentum=0.9)
        #self.opt = tf.train.AdagradOptimizer(lr)
 
        self.build_model(forward_only)

    def build_model(self, forward_only):
        print("[*] Building a PTRModel QA model")

        self.storys = tf.placeholder(tf.int32, [self.batch_size, self.story_size, self.input_dim], name='story')
        self.querys = tf.placeholder(tf.int32, [self.batch_size, self.input_dim], name='query')
        self.labels = tf.placeholder(tf.int32, [self.batch_size], name='label')

        self.embedding_matrix = weight('embedding', [self.output_dim, self.W], init='xavier')
        self.mask = tf.ones([self.input_dim, self.W]) #weight('mask', [self.input_dim, self.W], init='xavier')
        self.decoding_weight = weight('decoding_weight', [self.W, self.output_dim], init='xavier')
        self.decoding_bias = weight('decoding_bias', [self.output_dim], init='constant')

        zeros = np.zeros(self.W, dtype=np.float32)
        with tf.variable_scope(self.scope):
            init_state = self.controller.init_state()

            ss, qs = self.embedding(self.storys, self.querys)

            tf.get_variable_scope().reuse_variables()
            for i in range(self.batch_size):
                progress(i/float(self.batch_size))

                state = init_state
                for sid in range(self.story_size):
                    input_ = ss[i, sid:sid+1, :]
                    state = self.controller.update_memory(state, [input_, input_])

                # present inputs
                state['R'] = qs[i:i+1, :]

                outputs = []
                # present targets
                for _ in range(self.max_hops):
                    state = self.controller(state)
                    outputs.append(self.decode(state['R']))
                #out = tf.reduce_sum(tf.concat(outputs, 0), 0, keep_dims=True)
                out = outputs[-1]
                self.outputs.append(out)

            if not forward_only:
                logits = tf.concat(self.outputs, 0)
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
                self.loss = tf.reduce_mean(cross_entropy)

                predicts = tf.cast(tf.argmax(logits, 1), 'int32')
                corrects = tf.equal(predicts, self.labels)
                self.num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))

                if not self.params:
                    self.params = tf.trainable_variables()

                self.grads = []
                for grad in tf.gradients(self.loss, self.params):
                    if grad is not None:
                        self.grads.append(tf.clip_by_value(grad,
                                                      self.min_grad,
                                                      self.max_grad))
                    else:
                        self.grads.append(grad)

        with tf.variable_scope("opt", reuse=None):
            if not forward_only:
                self.optim = self.opt.apply_gradients(
                                  zip(self.grads, self.params),
                                  global_step=self.global_step)

        self.saver = tf.train.Saver()
        print(" [*] Build a PTRModel QA model finished")
 
    def get_outputs(self):
        return self.outputs

    def decode(self, ans):
        return tf.nn.relu(tf.nn.bias_add(tf.matmul(ans, self.decoding_weight), self.decoding_bias))

    def embedding(self, storys, querys):
        list = tf.unstack(storys)  # self.batch_size * [self.story_size, self.input_dim]
        embed_list = []
        for facts in list:
            facts = tf.unstack(facts) # self.story_size * self.input_dim
            embed = tf.stack([tf.nn.embedding_lookup(self.embedding_matrix, w) * self.mask for w in facts])  # [self.story_size, self.input_dim, self.W]
            embed_list.append(tf.reduce_sum(embed, 1)) # self.batch_size * [self.story_size, self.W]
        storys_embed = tf.stack(embed_list) # [self.batch_size, self.story_size, self.W]

        qs = tf.unstack(querys) # self.batch_size * self.input_dim
        embed = tf.stack([tf.nn.embedding_lookup(self.embedding_matrix, w) * self.mask for w in qs]) # [self.batch_size, self.input_dim, self.W]
        querys_embed = tf.reduce_sum(embed, 1) # [self.batch_size, self.W]

        return storys_embed, querys_embed

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

def main(_):
    print("Started Task:", FLAGS.task_id)

    train, test = load_task(FLAGS.data_dir, FLAGS.task_id)
    data = train + test

    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    max_story_size = max(list(map(len, (s for s, _, _ in data))))
    sentence_size = max(list(map(len, chain.from_iterable(s for s, _, _ in data))))
    query_size = max(list(map(len, (q for _, q, _ in data))))
    vocab_size = len(word_idx) + 1 # +1 for nil word
    sentence_size = max(query_size, sentence_size) # for the position

    S, Q, A = vectorize_data(train, word_idx, sentence_size, max_story_size)
    trainS, valS, trainQ, valQ, trainA, valA = cross_validation.train_test_split(S, Q, A, test_size=.1)
    testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, max_story_size)

    FLAGS.N = max(max_story_size, FLAGS.N)

    config = tf.ConfigProto()
    #config.graph_options.optimizer_options.opt_level=tf.OptimizerOptions.L3
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = PTRModel(sentence_size, vocab_size, FLAGS.N, FLAGS.W, FLAGS.G,
                    sess, batch_size=FLAGS.batch_size, story_size=max_story_size,
                    lr=FLAGS.learning_rate, epsilon=FLAGS.epsilon, 
                    controller_layer_size=FLAGS.controller_layer_size, 
                    controller_hidden_size=FLAGS.controller_hidden_size)

        print(" [*] Initalize all variables")
        #model.load(FLAGS.checkpoint_dir, 'QA_new')
        tf.initialize_all_variables().run()
        print(" [*] Initialization finished")

        start_time = time.time()
        if FLAGS.is_train:
            for t in range(FLAGS.epochs):
                total_corrects = 0.0
                total_cost = 0.0
                batchs = 0
                for start in range(0, trainS.shape[0], FLAGS.batch_size):
                    end = start + FLAGS.batch_size
                    feed_dict = {model.storys: trainS[start:end],
                                 model.querys: trainQ[start:end],
                                 model.labels: trainA[start:end]}

                    _, cost, corrects, step = sess.run([model.optim,
                                                        model.loss,
                                                        model.num_corrects,
                                                        model.global_step], feed_dict=feed_dict)
                    total_cost += cost
                    total_corrects += corrects
                    batchs += 1

                print("[epoch %5d]: train_loss:%.2f acc:%.2f (%.1fs)" % (t, 
                                                                         total_cost/batchs,
                                                                         total_corrects/(batchs*FLAGS.batch_size),
                                                                         time.time() - start_time))

                total_corrects = 0.0
                total_cost = 0.0
                batchs = 0
                for start in range(0, testS.shape[0], FLAGS.batch_size):
                    end = start + FLAGS.batch_size
                    feed_dict = {model.storys: testS[start:end],
                                 model.querys: testQ[start:end],
                                 model.labels: testA[start:end]}

                    cost, corrects = sess.run([model.loss, model.num_corrects], feed_dict=feed_dict)
                    total_cost += cost
                    total_corrects += corrects
                    batchs += 1

                print("[epoch %5d]: test_loss:%.2f acc:%.2f (%.1fs)" % (t, 
                                                                        total_cost/batchs, 
                                                                        total_corrects/(batchs*FLAGS.batch_size),
                                                                        time.time() - start_time))
                sys.stdout.flush()

                if t % 5 == 0 or t == FLAGS.epochs-1:
                    model.save(FLAGS.checkpoint_dir, 'QA', step)
        else:
            model.load(FLAGS.checkpoint_dir, 'QA')
            total_cost = 0.0
            total_corrects = 0.0
            batchs = 0
            for start in range(0, testS.shape[0], FLAGS.batch_size):
                end = start + FLAGS.batch_size
                feed_dict = {model.storys: testS[start:end],
                             model.querys: testQ[start:end],
                             model.labels: testA[start:end]}

                cost, corrects = sess.run([model.loss, model.num_corrects], feed_dict=feed_dict)
                total_cost += cost
                total_corrects += corrects
                batchs += 1
            print("Test cost:%.2f acc:%.2f (%.1fs)" % (total_cost/batchs, total_corrects/(batchs*FLAGS.batch_size), time.time() - start_time))

if __name__ == '__main__':
    tf.app.run()

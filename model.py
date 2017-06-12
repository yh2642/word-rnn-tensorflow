import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
import random
import numpy as np

from beam import BeamSearch

class Model():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cells = []
        for _ in range(args.num_layers):
            cell = cell_fn(args.rnn_size)
            cells.append(cell)

        self.cell = cell = rnn.MultiRNNCell(cells)

        self.input_data = tf.placeholder(tf.float32, [args.seq_length, args.batch_size, args.vocab_size])
        self.targets = tf.placeholder(tf.float32, [args.seq_length, args.batch_size, args.vocab_size])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        self.batch_pointer = tf.Variable(0, name="batch_pointer", trainable=False, dtype=tf.int32)
        self.inc_batch_pointer_op = tf.assign(self.batch_pointer, self.batch_pointer + 1)
        self.epoch_pointer = tf.Variable(0, name="epoch_pointer", trainable=False)
        self.batch_time = tf.Variable(0.0, name="batch_time", trainable=False)
        tf.summary.scalar("time_batch", self.batch_time)

        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                #with tf.name_scope('stddev'):
                #   stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                #tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                #tf.summary.histogram('histogram', var)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.sample_dim])
            variable_summaries(softmax_w)
            softmax_b = tf.get_variable("softmax_b", [args.sample_dim])
            variable_summaries(softmax_b)
            with tf.device("/cpu:0"):
                inputs = tf.split(self.input_data, args.batch_size, 0)
                inputs = [tf.squeeze(input_, [0]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            # prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return prev

        # outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=None if not infer else loop, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.probs, labels=self.targets)
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        tf.summary.scalar("cost", self.cost)
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, words, vocab, num=200, prime='first all', sampling_type=1, pick=0, width=4, tag_id_name_dict=dict(), tag_name_id_dict=dict()):
        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        def beam_search_predict(sample, state):
            """Returns the updated probability distribution (`probs`) and
            `state` for a given `sample`. `sample` should be a sequence of
            vocabulary labels, with the last word to be tested against the RNN.
            """

            x = np.zeros((1, 1))
            x[0, 0] = sample[-1]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, final_state] = sess.run([self.probs, self.final_state],
                                            feed)
            return probs, final_state

        def beam_search_pick(prime, width):
            """Returns the beam search pick."""
            if not len(prime) or prime == ' ':
                prime = random.choice(list(vocab.keys()))
            prime_labels = [vocab.get(word, 0) for word in prime.split()]
            bs = BeamSearch(beam_search_predict,
                            sess.run(self.cell.zero_state(1, tf.float32)),
                            prime_labels)
            samples, scores = bs.search(None, None, k=width, maxsample=num)
            return samples[np.argmin(scores)]

        def gen_vec(profile):
            vec = np.zeros(len(vocab))
            for cat_id, score in profile.items():
                idx = vocab[cat_id]
                vec[idx] = score
            return vec

        ret = ''
        if pick == 1:
            state = sess.run(self.cell.zero_state(1, tf.float32))
            if not len(prime) or prime == ' ':
                prime  = random.choice(list(vocab.keys()))
                print prime
            for word in prime[:-1]:
                # print '-'.join(map(lambda cat_id: unicode(tag_id_name_dict.get(cat_id, cat_id)), word.keys()))
                print '-'.join(word.keys())
                x = gen_vec(word)
                feed = {self.input_data: x, self.initial_state:state}
                [state] = sess.run([self.final_state], feed)
            print state
            #
            # for n in range(num):
            #     x = np.zeros((1, 1))
            #     x[0, 0] = vocab.get(word, 0)
            #     feed = {self.input_data: x, self.initial_state:state}
            #     [probs, state] = sess.run([self.probs, self.final_state], feed)
            #     p = probs[0]
            #
            #     if sampling_type == 0:
            #         sample = np.argmax(p)
            #     elif sampling_type == 2:
            #         if word == '\n':
            #             sample = weighted_pick(p)
            #         else:
            #             sample = np.argmax(p)
            #     else: # sampling_type == 1 default:
            #         sample = weighted_pick(p)
            #     if n==1:
            #         top_interests = np.argsort(-p)[:20]
            #         for idx in top_interests:
            #             print tag_id_name_dict.get(words[idx], words[idx]), p[idx]
            #     pred = words[sample]
            #     ret += ' ' + tag_id_name_dict.get(pred, pred)
            #     word = pred
        elif pick == 2:
            pred = beam_search_pick(prime, width)
            for i, label in enumerate(pred):
                ret += ' ' + words[label] if i > 0 else words[label]
        return ret

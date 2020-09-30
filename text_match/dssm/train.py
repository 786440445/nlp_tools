import os
import sys
home_dir = os.getcwd()
sys.path.append(home_dir)

import tensorflow as tf
from text_match.dssm.graph import Graph
from text_match.dssm import args
from text_match.utils.load_data import load_char_data

p, h, y = load_char_data('train.txt', data_size=None)
p_eval, h_eval, y_eval = load_char_data('dev.txt', data_size=None)

p_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='p')
h_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='h')
y_holder = tf.placeholder(dtype=tf.int32, shape=None, name='y')

dataset = tf.data.Dataset.from_tensor_slices((p_holder, h_holder, y_holder))
dataset = dataset.batch(args.batch_size).repeat(args.epochs)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

model = Graph()
saver = tf.train.Saver(max_to_keep=3)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer, feed_dict={p_holder: p, h_holder: h, y_holder: y})
    steps = int(len(y) / args.batch_size)
    old_eval_acc = 0
    for epoch in range(args.epochs):
        for step in range(steps):
            p_batch, h_batch, y_batch = sess.run(next_element)
            _, loss, acc = sess.run([model.train_op, model.loss, model.acc],
                                    feed_dict={model.p: p_batch,
                                               model.h: h_batch,
                                               model.y: y_batch,
                                               model.keep_prob: args.keep_prob})
            print('epoch:', epoch, ' step:', step, ' loss: ', loss, ' acc:', acc)

        loss_eval, acc_eval = sess.run([model.loss, model.acc],
                                       feed_dict={model.p: p_eval,
                                                  model.h: h_eval,
                                                  model.y: y_eval,
                                                  model.keep_prob: 1})
        print('loss_eval: ', loss_eval, ' acc_eval:', acc_eval)
        print('\n')
        if acc_eval > old_eval_acc:
            saver.save(sess, os.path.join(home_dir, 'text_match/output/dssm/dssm_%d_%.4f.ckpt' % (epoch, acc_eval)))
            saver.save(sess, os.path.join(home_dir, 'text_match/output/dssm/final_model.ckpt'))
            old_eval_acc = acc_eval
        else:
            print('not improved')
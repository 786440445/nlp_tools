import tensorflow as tf


a = tf.constant([1,0,1,0,0])
[0,1]
[1,0]
y = tf.one_hot(a, 2)
with tf.Session() as sess:
    print(sess.run(y))

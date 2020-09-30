import tensorflow as tf


class Graph:
    def __init__(self, args):
        self.embedding_size = args.embedding_size
        # self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.vocab_size = args.vocab_size
        self.is_training = args.is_training
        self.class_size = args.class_size
        self.learning_rate = args.learning_rate

        self.filter_sizes_1 = '3, 4, 5'
        self.filter_sizes_2 = '3, 4, 5'
        self.num_filters = 64
        self.initializer = None

        self.init_placeholder()
        self.forward()

    def init_placeholder(self):
        self.q_text = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length])
        self.a_text = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length])
        self.label = tf.placeholder(dtype=tf.int32, shape=None)
        self.batch_size = tf.placeholder(dtype=tf.int32, name='batch_size')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')
        self.embedding_table = tf.get_variable(dtype=tf.float32, shape=(self.vocab_size, self.embedding_size),
                                               name='embedding')

    def forward(self):
        self.q_embedding = tf.nn.embedding_lookup(self.embedding_table, self.q_text)
        self.a_embedding = tf.nn.embedding_lookup(self.embedding_table, self.a_text)

        self.Q = self.encode(self.q_embedding)  # b * m * c
        self.A = self.encode(self.a_embedding)  # b * n * c
        self.r_q, self.r_a = self.attention_pooling(self.Q, self.A)
        self.score = self._cosine(self.r_q, self.r_a)
        # self._model_stats()  # print model statistics info
        self.score = tf.expand_dims(self.score, -1)
        neg_result = 1 - self.score
        # print(self.score)
        # print(neg_result)
        logits = tf.concat([neg_result, self.score], axis=1)
        self.train(logits)


    def encode(self, x):
        conv1_outputs = []
        for i, filter_size in enumerate(map(int, self.filter_sizes_1.split(','))):
            with tf.variable_scope("1st_conv_{}".format(filter_size), reuse=tf.AUTO_REUSE):
                filter_shape = [filter_size, self.embedding_size, self.num_filters]
                W = tf.get_variable("first_conv_{}_W".format(filter_size), shape=filter_shape, initializer=self.initializer)
                conv1 = tf.nn.conv1d(x, filters=W, stride=1, padding="SAME", name="first_conv")
                conv1 = tf.layers.batch_normalization(conv1, axis=-1, training=self.is_training)  # axis定的是channel在的维度。
                h = tf.nn.relu(conv1, name="relu_1")
                conv1_outputs.append(h)

        conv2_inputs = tf.concat(conv1_outputs, -1)

        conv2_outputs = []
        for i, filter_size in enumerate(map(int, self.filter_sizes_2.split(','))):
            with tf.variable_scope("second_conv_maxpool_%s" % filter_size, reuse=tf.AUTO_REUSE):
                # Convolution Layer
                filter_shape = [filter_size, self.num_filters * len(self.filter_sizes_1.split(',')), self.num_filters]
                W = tf.get_variable("second_conv_{}_W".format(filter_size), shape=filter_shape, initializer=self.initializer)
                conv2 = tf.nn.conv1d(conv2_inputs, W, stride=1, padding="SAME", name="second_conv")
                conv2 = tf.layers.batch_normalization(conv2, axis=-1, training=self.is_training)  # axis定的是channel在的维度。
                h = tf.nn.relu(conv2, name="relu_2")
                conv2_outputs.append(h)
        outputs = tf.concat(conv2_outputs, 2, name="output")  # (batch_size, seq_length, num_filters_total)
        if self.is_training:
            outputs = tf.nn.dropout(outputs, self.keep_prob, name="output")
        return outputs

    def attention_pooling(self, q, a):
        hidden_size = q.get_shape().as_list()[-1]  # vector size
        with tf.variable_scope('attention_pooling', reuse=tf.AUTO_REUSE):
            self.Q = q
            self.A = a
            self.U = tf.get_variable('U', [hidden_size, hidden_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.U_batch = tf.tile(tf.expand_dims(self.U, 0), [self.batch_size, 1, 1])
            self.G = tf.tanh(
                tf.matmul(
                    tf.matmul(self.Q, self.U_batch), tf.transpose(self.A, [0, 2, 1]))
            )  # G b*m*n
            # column-wise and row-wise max-poolings to generate g_q (b*m*1), g_a (b*1*n)
            g_q = tf.reduce_max(self.G, axis=2, keepdims=True)
            g_a = tf.reduce_max(self.G, axis=1, keepdims=True)
            sigma_q = tf.nn.softmax(g_q)
            sigma_a = tf.nn.softmax(g_a)
            r_q = tf.squeeze(tf.matmul(tf.transpose(self.Q, [0, 2, 1]), sigma_q), axis=2)
            r_a = tf.squeeze(tf.matmul(sigma_a, self.A), axis=1)
            return r_q, r_a

    def train(self, logits):
        y = tf.one_hot(self.label, self.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        prediction = tf.argmax(logits, axis=1)
        correct_prediction = tf.equal(tf.cast(prediction, tf.int32), self.label)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    @staticmethod
    def _cosine(x, y):
        """x, y shape (batch_size, vector_size)"""
        cosine = tf.div(
            tf.reduce_sum(x * y, 1),
            tf.sqrt(tf.reduce_sum(x * x, 1)) * tf.sqrt(tf.reduce_sum(y * y, 1)) + 1e-8,
            name="cosine")
        return cosine

    @staticmethod
    def _model_stats():
        """Print trainable variables and total model size."""

        def size(v):
            return tf.reduce_all(lambda x, y: x * y, v.get_shape().as_list())

        print("Trainable variables")
        for v in tf.trainable_variables():
            print("  %s, %s, %s, %s" % (v.name, v.device, str(v.get_shape()), size(v)))
        print("Total model size: %d" % (sum(size(v) for v in tf.trainable_variables())))
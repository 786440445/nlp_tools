import tensorflow as tf

class Graph:
    def __init__(self, args):
        self.embedding_size = args.embedding_size
        self.max_length = args.max_len
        self.vocab_size = args.vocab_size
        self.is_training = args.is_training
        self.class_size = args.class_size
        self.learning_rate = args.learning_rate
        self.filter_sizes_1 = '3, 4, 5'
        self.num_filters = 128
        self.initializer = None
        self.filter_sizes = [3, 4, 5]
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
        self.kernel_size = 3
        self.init_placeholder()
        self.build_model()

    def init_placeholder(self):
        self.seq = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length])
        self.label = tf.placeholder(dtype=tf.int32, shape=[None, self.class_size])
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')
        self.embedding_table = tf.get_variable(dtype=tf.float32, shape=(self.vocab_size, self.embedding_size),
                                               name='embedding')

    def build_model(self):
        # 词向量映射
        with tf.name_scope("embedding"):
            embedding_inputs = tf.nn.embedding_lookup(self.embedding_table, self.seq)
            embedding_inputs = tf.expand_dims(embedding_inputs, axis=-1)  # [None,seq,embedding,1]
            # region_embedding  # [batch,seq-3+1,1,250]
            region_embedding = tf.layers.conv2d(embedding_inputs, self.num_filters,
                                                [self.kernel_size, self.embedding_size])

            pre_activation = tf.nn.relu(region_embedding, name='preactivation')
            # [B,L,200, 1]

        with tf.name_scope("conv3_0"):
            # [B,L,200, 128]
            conv3 = tf.layers.conv2d(pre_activation, self.num_filters, self.kernel_size,
                                     padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(conv3)

        with tf.name_scope("conv3_1"):
            # [B,L,200, 128]
            conv3 = tf.layers.conv2d(conv3, self.num_filters, self.kernel_size,
                                     padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(conv3)

        # resdul
        conv3 = self.squeeze_excitation_layer(conv3, out_dim=128, ratio=2) + region_embedding
        with tf.name_scope("pool_1"):
            pool = tf.pad(conv3, paddings=[[0, 0], [0, 1], [0, 0], [0, 0]])
            pool = tf.nn.max_pool(pool, [1, 3, 1, 1], strides=[1, 2, 1, 1], padding='VALID')

        with tf.name_scope("conv3_2"):
            conv3 = tf.layers.conv2d(pool, self.num_filters, self.kernel_size,
                                     padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(conv3)

        with tf.name_scope("conv3_3"):
            conv3 = tf.layers.conv2d(conv3, self.num_filters, self.kernel_size,
                                     padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(conv3)

        # resdul
        conv3 = self.squeeze_excitation_layer(conv3, out_dim=128, ratio=1) + pool
        pool_size = int((self.max_length - 3 + 1) / 2)
        conv3 = tf.layers.max_pooling1d(tf.squeeze(conv3, [2]), pool_size, 1)
        conv3 = tf.squeeze(conv3, [1])  # [batch,250]
        conv3 = tf.nn.dropout(conv3, self.keep_prob)

        with tf.name_scope("score"):
            # classify
            self.logits = tf.layers.dense(conv3, self.class_size, name='fc2')
            self.preds = tf.nn.sigmoid(self.logits)

        with tf.name_scope("loss"):
            # 损失函数，交叉熵
            # print(self.label)
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits, labels=tf.cast(self.label, tf.float32))

            l2_loss = tf.losses.get_regularization_loss()
            self.loss = tf.reduce_mean(cross_entropy, name="loss")
            self.loss += l2_loss
            # optim
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def squeeze_excitation_layer(self, input_x, out_dim, ratio):
        squeeze = tf.reduce_mean(input_x, [1, 2], keep_dims=True)
        excitation = tf.layers.dense(squeeze, units=out_dim / ratio, activation='relu')
        excitation = tf.layers.dense(excitation, units=out_dim, activation='sigmoid')
        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = tf.multiply(input_x, excitation)
        return scale






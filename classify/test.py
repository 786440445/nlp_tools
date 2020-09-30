import os, sys
home_dir = os.getcwd()
sys.path.append(home_dir)

import pandas as pd
import tensorflow as tf

from classify import args
from classify.data_loader import padding, sentence2id, load_char_vocab
from classify.DPCNN import Graph


def load_char_data(filename):
    word2idx, idx2word = load_char_vocab()
    test_df = pd.read_csv(filename, encoding='utf-8')
    id_list = test_df["ID"].values.tolist()
    sen_list = test_df["Question Sentence"].values.tolist()
    sen_list = padding(sen_list, max_len=300)
    sen_list = sentence2id(sen_list, word2idx)
    return id_list, sen_list


if __name__ == '__main__':
    id_list, sen_list = load_char_data('classify/data/nlp_test.csv')
    seq_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.max_len), name='seq_holder')

    dataset = tf.data.Dataset.from_tensor_slices(seq_holder)
    dataset_test = dataset.batch(100)
    iterator = dataset_test.make_initializable_iterator()
    next_element = iterator.get_next()

    model = Graph(args)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    out = []
    with tf.Session() as sess:
        model_path = 'classify/model'
        ckpt = tf.train.latest_checkpoint(model_path)
        saver.restore(sess, ckpt)
        sess.run(iterator.initializer, feed_dict={seq_holder: sen_list})
        test_step = len(sen_list)//100
        for seq in range(test_step):
            seq_batch = sess.run(next_element)
            feed_dict = {model.seq: seq_batch,
                         model.keep_prob: 1}
            preds = sess.run([model.preds], feed_dict=feed_dict)
            out.extend(preds[0])

    assert len(out) == len(id_list)
    csv_out = []
    for pred, id in zip(out, id_list):
        label = []
        print(pred)
        for index, value in enumerate(pred):
            if value >= 0.5:
                label.append(1)
            else:
                label.append(0)
        label = tuple([id] + label)
        csv_out.append(label)
    csv_out = pd.DataFrame(data=csv_out, columns=["ID", "category_A", "category_B", "category_C",
                                                  "category_D", "category_E", "category_F"])
    csv_out.to_csv('classify/data/submit.csv', encoding='utf-8', index=False)
import os
import sys
home_dir = os.getcwd()
sys.path.append(home_dir)

import numpy as np
import tensorflow as tf
# from classify.graph import Graph
from classify.DPCNN import Graph

from classify import args
from classify.data_loader import load_char_data


def evaluate(sess, model, x_, y_):
    """
    评估 val data 的准确率和损失
    """
    y_pred = []
    y_target = []
    feed_dict = {model.seq: x_, model.label: y_, model.keep_prob: 1}
    loss, logits = sess.run([model.loss, model.logits], feed_dict=feed_dict)
    y_batch_pred = get_logits_label(logits)
    # print('y_batch_pred', y_batch_pred)
    y_batch_target = get_target_label(y_)
    # print('y_batch_target', y_batch_target)
    y_pred.extend(y_batch_pred)
    y_target.extend(y_batch_target)
    confuse_matrix = compute_confuse_matrix(y_target, y_pred)
    f1_micro, f1_macro = compute_micro_macro(confuse_matrix)
    # print(f1_micro, f1_macro)
    f1_score = (f1_micro + f1_macro) / 2.0

    return loss, f1_score, confuse_matrix, y_pred, y_target


def get_logits_label(logits):
    y_predict_labels = []
    for line in logits:
        line_label = [i for i in range(len(line)) if line[i] >= 0.50]
        y_predict_labels.append(line_label)
    return y_predict_labels


def get_target_label(eval_y):
    eval_y_short = []
    for line in eval_y:
        target = []
        for index, label in enumerate(line):
            if label > 0:
                target.append(index)
        eval_y_short.append(target)
    return eval_y_short


def compute_confuse_matrix(target_y, predict_y):
    """
    compute TP, FP, FN given target label and predict label
    :param target_y:
    :param predict_y:
    :param label_dict {label:(TP,FP,FN)}
    :return: macro_f1(a scalar), micro_f1(a scalar)
    """
    # count number of TP,FP,FN for each class
    label_dict = {}
    for i in range(6):
        label_dict[i] = (0, 0, 0)

    for num in range(len(target_y)):
        targe_tmp = target_y[num]
        pre_tmp = predict_y[num]
        unique_labels = set(targe_tmp + pre_tmp)
        for label in unique_labels:
            TP, FP, FN = label_dict[label]
            if label in pre_tmp and label in targe_tmp:  # predict=1,truth=1 (TP)
                TP = TP + 1
            elif label in pre_tmp and label not in targe_tmp:  # predict=1,truth=0(FP)
                FP = FP + 1
            elif label not in pre_tmp and label in targe_tmp:  # predict=0,truth=1(FN)
                FN = FN + 1
            label_dict[label] = (TP, FP, FN)
    print(label_dict)
    return label_dict


def compute_micro_macro(label_dict):
    f1_micro = compute_f1_micro(label_dict)
    f1_macro = compute_f1_macro(label_dict)
    return f1_micro, f1_macro


def compute_f1_micro(label_dict):
    """
    compute f1_micro
    :param label_dict: {label:(TP,FP,FN)}
    :return: f1_micro: a scalar
    """
    TP_micro, FP_micron, FN_micro = compute_micro(label_dict)
    f1_micro = compute_f1(TP_micro, FP_micron, FN_micro)
    return f1_micro


def compute_f1(TP, FP, FN, small_value=0.00001):
    """
    compute f1
    :param TP_micro: number.e.g. 200
    :param FP_micro: number.e.g. 200
    :param FN_micro: number.e.g. 200
    :return: f1_score: a scalar
    """
    precison = TP / (TP + FP + small_value)
    recall = TP / (TP + FN + small_value)
    f1_score = (2 * precison * recall) / (precison + recall + small_value)

    return f1_score


def compute_f1_macro(label_dict):
    """
    compute f1_macro
    :param label_dict: {label:(TP,FP,FN)}
    :return: f1_macro
    """
    f1_dict = {}
    num_classes = len(label_dict)
    for label, tuplee in label_dict.items():
        TP, FP, FN = tuplee
        f1_score_onelabel = compute_f1(TP, FP, FN)
        f1_dict[label] = f1_score_onelabel
    f1_score_sum = 0.0
    for label, f1_score in f1_dict.items():
        f1_score_sum += f1_score
    f1_score = f1_score_sum / float(num_classes)
    return f1_score


def compute_micro(label_dict):
    """
    compute micro FP,FP,FN
    :param label_dict_accusation: a dict. {label:(TP, FP, FN)}
    :return:TP_micro,FP_micro,FN_micro
    """
    TP_micro, FP_micro, FN_micro = 0.0, 0.0, 0.0
    for label, tuplee in label_dict.items():
        TP, FP, FN = tuplee
        TP_micro = TP_micro + TP
        FP_micro = FP_micro + FP
        FN_micro = FN_micro + FN
    return TP_micro, FP_micro, FN_micro


if __name__ == '__main__':
    seqs, labels = load_char_data('train.csv')
    length = len(seqs)
    train_len = 4*length//5
    dev_len = length//5

    shuffle_list = np.random.permutation(length)
    shuffle_seqs = seqs[shuffle_list]
    shuffle_labels = labels[shuffle_list]

    seqs_train = shuffle_seqs
    labels_train = shuffle_labels

    seqs_dev = shuffle_seqs[train_len:]
    labels_dev = shuffle_labels[train_len:]

    print('训练样本长度: ', len(seqs_train))
    print('验证集本长度: ', len(seqs_dev))

    seq_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.max_len), name='seq_holder')
    label_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.class_size), name='label_holder')

    dataset = tf.data.Dataset.from_tensor_slices((seq_holder, label_holder))
    dataset_train = dataset.batch(args.batch_size).repeat(args.epochs)
    iterator = dataset_train.make_initializable_iterator()
    next_element = iterator.get_next()

    model = Graph(args)
    saver = tf.train.Saver(max_to_keep=3)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    with tf.Session(config=config)as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer, feed_dict={seq_holder: seqs_train, label_holder: labels_train})
        steps = int(len(seqs_train) / args.batch_size)
        old_f1 = 0
        total_loss = 0
        train_step = 0
        for epoch in range(args.epochs):
            model.is_training = True
            for step in range(steps):
                train_step += 1
                seq_batch, label_batch = sess.run(next_element)
                _, loss = sess.run([model.train_op, model.loss],
                                   feed_dict={model.seq: seq_batch,
                                              model.label: label_batch,
                                              model.keep_prob: args.keep_prob})
                total_loss += loss
                print('epoch : %d train_loss: %.4f' % (epoch, total_loss / train_step))
                if train_step % 5 == 0:
                    val_loss, val_f1, confuse_matrix, y_pred, y_target = evaluate(sess, model, seqs_dev, labels_dev)
                    count = 0
                    for pred, target in zip(y_pred, y_target):
                        if pred == target:
                            count += 1
                    acc = count/len(y_pred)
                    print('loss_eval: %.4f val_loss: %.4f, acc: %.4f val_f1: %.4f' % (total_loss / train_step,
                                                                                      val_loss, acc, val_f1))
            if val_f1 > old_f1:
                saver.save(sess, os.path.join(home_dir, 'classify/model/epoch{}_acc{:.4f}_f1_{:.4f}.ckpt'.
                                              format(epoch, acc, val_f1)))
                old_f1 = val_f1
            else:
                print('not improved')

import tensorflow as tf
import numpy as np
from PIL import Image

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_LEN = IMAGE_WIDTH * IMAGE_HEIGHT
HIDEN_SIZE = 100
OUTPUT = 10
DROP_OUT = 1.0
TRAIN_LINK = "/home/wxx/data/num_train/link.txt"
TEST_LINK = "/home/wxx/data/num_test/link.txt"

def dataAn(data):
    datas = data.split()
    return datas[0], int(datas[1])
    pass

def loadData(file):
    # datas = []
    datas = np.zeros((0, IMAGE_LEN))
    labels = np.zeros((0, OUTPUT))
    with open(file, 'r') as f:
        data = f.readlines()
        for line in data:
            label = [0 for _ in range(OUTPUT)]
            image_name, label_index = dataAn(line)
            
            label[label_index] = 1
            labels = np.vstack((labels, label))

            image = np.array(Image.open(image_name))
            datas = np.vstack((datas, image.reshape(1, IMAGE_LEN)[0]))
    
    return datas, labels
    pass

# 使用优化器
def main():
    x = tf.placeholder(tf.float32, shape=(None, IMAGE_LEN))
    y = tf.placeholder(tf.float32, shape=(None, OUTPUT))
    keep_prob = tf.placeholder(tf.float32)
    lr = tf.Variable(0.001, dtype=tf.float32)

    # w1 = tf.Variable(tf.zeros([IMAGE_LEN, OUTPUT]))
    w1 = tf.Variable(tf.truncated_normal([IMAGE_LEN, HIDEN_SIZE], stddev=0.1))
    b1 = tf.Variable(tf.zeros([HIDEN_SIZE]) + 0.1)
    L1 = tf.nn.tanh(tf.matmul(x, w1) + b1)
    L1_drop = tf.nn.dropout(L1, keep_prob)

    w2 = tf.Variable(tf.truncated_normal([HIDEN_SIZE, OUTPUT], stddev=0.1))
    b2 = tf.Variable(tf.zeros([OUTPUT]) + 0.1)


    prediction = tf.nn.softmax(tf.matmul(L1_drop, w2) + b2)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

    # 交叉熵代价函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

    # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.global_variables_initializer()

    train_datas, train_labels = loadData(TRAIN_LINK)
    test_datas, test_labels = loadData(TEST_LINK)

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(21000):
            sess.run(tf.assign(lr, 0.001 * (0.9999 ** epoch)))
            sess.run(train_step, feed_dict={x:train_datas, y:train_labels, keep_prob:DROP_OUT})

            acc = sess.run(accuracy, feed_dict={x:test_datas, y:test_labels, keep_prob:DROP_OUT})
            train_acc = sess.run(accuracy, feed_dict={x:train_datas, y:train_labels, keep_prob:DROP_OUT})
            learing_rate = sess.run(lr)
            print ("[%d] accuracy : %f; train accuracy : %f; learing_rate : %f"%(epoch, acc, train_acc, learing_rate))
    pass

# 包含隐含层
# def main():
#     x = tf.placeholder(tf.float32, shape=(None, IMAGE_LEN))
#     y = tf.placeholder(tf.float32, shape=(None, OUTPUT))
#     keep_prob = tf.placeholder(tf.float32)

#     # w1 = tf.Variable(tf.zeros([IMAGE_LEN, OUTPUT]))
#     w1 = tf.Variable(tf.truncated_normal([IMAGE_LEN, HIDEN_SIZE], stddev=0.1))
#     b1 = tf.Variable(tf.zeros([HIDEN_SIZE]) + 0.1)
#     L1 = tf.nn.tanh(tf.matmul(x, w1) + b1)
#     L1_drop = tf.nn.dropout(L1, keep_prob)

#     w2 = tf.Variable(tf.truncated_normal([HIDEN_SIZE, OUTPUT], stddev=0.1))
#     b2 = tf.Variable(tf.zeros([OUTPUT]) + 0.1)


#     prediction = tf.nn.softmax(tf.matmul(L1_drop, w2) + b2)
#     correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

#     train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     init = tf.global_variables_initializer()

#     train_datas, train_labels = loadData(TRAIN_LINK)
#     test_datas, test_labels = loadData(TEST_LINK)

#     with tf.Session() as sess:
#         sess.run(init)
#         for epoch in range(21000):
#             sess.run(train_step, feed_dict={x:train_datas, y:train_labels, keep_prob:DROP_OUT})

#             acc = sess.run(accuracy, feed_dict={x:test_datas, y:test_labels, keep_prob:DROP_OUT})
#             train_acc = sess.run(accuracy, feed_dict={x:train_datas, y:train_labels, keep_prob:DROP_OUT})

#             print ("[%d] accuracy : %f, train accuracy : %f"%(epoch, acc, train_acc))
#     pass

# 交叉熵测试
# def main():
#     x = tf.placeholder(tf.float32, shape=(None, IMAGE_LEN))
#     y = tf.placeholder(tf.float32, shape=(None, OUTPUT))

#     # w = tf.Variable(tf.random_normal([IMAGE_LEN, OUTPUT], stddev=0.1))
#     w = tf.Variable(tf.zeros([IMAGE_LEN, OUTPUT]))
#     b = tf.Variable(tf.zeros([OUTPUT]))

#     prediction = tf.nn.softmax(tf.matmul(x, w)+b)
#     correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

#     # loss = tf.reduce_mean(tf.square(y - prediction))
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

#     train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     init = tf.global_variables_initializer()

#     train_datas, train_labels = loadData(TRAIN_LINK)
#     test_datas, test_labels = loadData(TEST_LINK)

#     with tf.Session() as sess:
#         sess.run(init)
#         for epoch in range(21000):
#             sess.run(train_step, feed_dict={x:train_datas, y:train_labels})

#             acc = sess.run(accuracy, feed_dict={x:test_datas, y:test_labels})
#             train_acc = sess.run(accuracy, feed_dict={x:train_datas, y:train_labels})

#             print ("[%d] accuracy : %f, train accuracy : %f"%(epoch, acc, train_acc))
#     pass

# def main():
#     x = tf.placeholder(tf.float32, shape=(None, IMAGE_LEN))
#     y = tf.placeholder(tf.float32, shape=(None, OUTPUT))

#     # w = tf.Variable(tf.random_normal([IMAGE_LEN, OUTPUT], stddev=0.1))
#     w = tf.Variable(tf.zeros([IMAGE_LEN, HIDEN_SIZE]))
#     b = tf.Variable(tf.zeros([HIDEN_SIZE]))

#     w2 = tf.Variable(tf.zeros([HIDEN_SIZE, OUTPUT]))
#     b2 = tf.Variable(tf.zeros([OUTPUT]))

#     prediction = tf.nn.softmax(tf.matmul(x, w)+b)
#     correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

#     loss = tf.reduce_mean(tf.square(y - prediction))

#     train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     init = tf.global_variables_initializer()

#     train_datas, train_labels = loadData(TRAIN_LINK)
#     test_datas, test_labels = loadData(TEST_LINK)

#     with tf.Session() as sess:
#         sess.run(init)
#         for epoch in range(21000):
#             sess.run(train_step, feed_dict={x:train_datas, y:train_labels})

#             acc = sess.run(accuracy, feed_dict={x:test_datas, y:test_labels})

#             print ("[%d] accuracy : %f"%(epoch, acc))
#     pass

if __name__ == '__main__':
    main()

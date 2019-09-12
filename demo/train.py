#coding:utf-8

import tensorflow as tf
import numpy as np
from PIL import Image

# 参数设置
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_LEN = IMAGE_WIDTH * IMAGE_HEIGHT
OUTPUT = 10
DROP_OUT = 1.0
TRAIN_LINK = "../data/num_train/link.txt"
TEST_LINK = "../data/num_test/link.txt"
SAVE_MODEL = "../num_model/num_model"
UPDATE_LOOP = 1        # 更新测试训练准确率的频率
SAVE_LOOP = 200         # 保存模型的频率

# 解析一行数据
def dataAn(data):
    datas = data.split()
    return datas[0], int(datas[1])

# 加载数据
def loadData(file):

    # 初始化返回参数
    datas = np.zeros((0, IMAGE_LEN), dtype=float)
    labels = np.zeros((0, OUTPUT), dtype=float)

    # 解析文件, 添加数据
    with open(file, 'r') as f:
        data = f.readlines()
        for line in data:
            # 标签初始化
            label = [0 for _ in range(OUTPUT)]

            image_name, label_index = dataAn(line)

            # 添加标签数据
            label[label_index] = 1
            labels = np.vstack((labels, label))

            # 转换并添加图像数据
            image = np.array(Image.open(image_name))
            datas = np.vstack((datas, image.reshape(1, IMAGE_LEN)[0]))
    
    return datas, labels

# 新建权重
def weightVariabel(shape):
    initial = tf.random.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

# 新建
def biasVariable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# 卷集层
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

# 池化
def maxPool2_2(x):
    return tf.nn.max_pool2d(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 网络
def net():
    # x, y, prediction = net()
    # 创建占位符号, 用于输入训练集, None表示数据量不确定, name不要忘记, 使用模型的时候要用到
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, IMAGE_LEN], name='inputs')
    y = tf.compat.v1.placeholder(tf.float32, shape=[None, OUTPUT], name="labels")

    # 输入图像格式转换
    x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])

    # 设置第一层卷集权重和偏移, 输入为一张图片
    w_conv1 = weightVariabel([5, 5, 1, 32])
    b_conv1 = biasVariable([32])

    # 卷集和池化
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = maxPool2_2(h_conv1)

    # 设置第二层卷集权重和偏移, 输入为上面池化输出的16个数据
    w_conv2 = weightVariabel([5, 5, 32, 64])
    b_conv2 = biasVariable([64])

    # 卷集和池化
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = maxPool2_2(h_conv2)

    # 
    w_fc1 = weightVariabel([7 * 7 * 64, 128])
    b_fc1 = biasVariable([128])

    # 
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64]) 
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
    
    # 输出层参数
    w_fc2 = weightVariabel([128, OUTPUT])
    b_fc2 = biasVariable([OUTPUT])

    # 
    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 预测, 不要忘记name, 使用模型时要用
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2, name='prediction')

    return x, y, prediction, keep_prob

def main():
    x, y, prediction, keep_prob = net()

    # 交叉熵
    cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction)) 

    # 训练优化方案和训练方向, 使用Adam, 向交叉熵减小的方向训练
    train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # 预测结果对比, 计算准确率
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 加载训练集和测试集, 如果数据量太大, 可以分批训练
    train_datas, train_labels = loadData(TRAIN_LINK)
    test_datas, test_labels = loadData(TEST_LINK)

    test_accuracy = 0.0
    # 开始训练
    with tf.compat.v1.Session() as sess:
        # 用于保存模型
        saver = tf.compat.v1.train.Saver()

        # 初始化变量
        sess.run(tf.compat.v1.global_variables_initializer())

        for epoch in range(1, 2001):
            # 训练
            sess.run(train_step, feed_dict={x:train_datas, y:train_labels, keep_prob:0.7})

            if epoch % UPDATE_LOOP == 0:
                ## 计算训练和测试的准确率
                train_accuracy = sess.run(accuracy, feed_dict={x:train_datas, y:train_labels, keep_prob:1})
                test_accuracy = sess.run(accuracy, feed_dict={x:test_datas, y:test_labels, keep_prob:1})
                print ("[%d] train_accuracy : %f; test_accuracy : %f;"%(epoch, train_accuracy, test_accuracy))

            if epoch % SAVE_LOOP == 0 and test_accuracy > 0.9:
                saver.save(sess, SAVE_MODEL, global_step=epoch)


if __name__ == '__main__':
    main()
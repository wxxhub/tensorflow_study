import tensorflow as tf
import numpy as np
from PIL import Image

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_LEN = IMAGE_WIDTH * IMAGE_HEIGHT
OUTPUT = 10
TRAIN_TF = "train.tfrecords"
TEST_TF = "test.tfrecords"

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
            label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            image_name, label_index = dataAn(line)
            label[label_index] = 1
            labels = np.vstack((labels, label))
            image = np.array(Image.open(image_name))
            # datas.append(image.reshape(1, IMAGE_LEN)[0])
            datas = np.vstack((datas, image.reshape(1, IMAGE_LEN)[0]))
    
    return datas, labels
    pass

def main():
    x = tf.placeholder(tf.float32, shape=(None, IMAGE_LEN))
    y = tf.placeholder(tf.float32, shape=(None, OUTPUT))

    w = tf.Variable(tf.zeros([IMAGE_LEN, OUTPUT]))
    b = tf.Variable(tf.zeros([OUTPUT]))

    prediction = tf.nn.softmax(tf.matmul(x, w)+b)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

    loss = tf.reduce_mean(tf.square(y - prediction))

    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.global_variables_initializer()

    train_datas, train_labels = loadData("/home/wxx/data/num_train/link.txt")
    test_datas, test_labels = loadData("/home/wxx/data/num_test/link.txt")

    print (train_labels)
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(210):
            sess.run(train_step, feed_dict={x:train_datas, y:train_labels})

            acc = sess.run(accuracy, feed_dict={x:test_datas, y:test_labels})

            print ("[%d] accuracy : %f"%(epoch, acc))
    pass

if __name__ == '__main__':
    main()

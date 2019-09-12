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
    datas = np.zeros((0, IMAGE_LEN), dtype=float)
    labels = np.zeros((0, OUTPUT), dtype=float)
    with open(file, 'r') as f:
        data = f.readlines()
        for line in data:
            label = [0 for _ in range(OUTPUT)]
            image_name, label_index = dataAn(line)
            
            label[label_index] = float(1)
            labels = np.vstack((labels, label))

            image = np.array(Image.open(image_name))
            datas = np.vstack((datas, image.reshape(1, IMAGE_LEN)[0]))
    
    return datas, labels
    pass

def weightVariabel(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def biasVariable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def maxPool2_2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def main():
    x = tf.placeholder(tf.float32, shape=(None, IMAGE_LEN), name='inputs')
    y = tf.placeholder(tf.float32, shape=(None, OUTPUT), name='labels')

    x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])

    w_conv1 = weightVariabel([5, 5, 1, 32])
    b_conv1 = biasVariable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = maxPool2_2(h_conv1)

    w_conv2 = weightVariabel([5, 5, 32, 64])
    b_conv2 = biasVariable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = maxPool2_2(h_conv2)

    w_fc1 = weightVariabel([7 * 7 * 64, 50])
    b_fc1 = biasVariable([50])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64]) 
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    w_fc2 = weightVariabel([50, 10])
    b_fc2 = biasVariable([10])

    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2, name='prediction')

    cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction)) 
    

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_datas, train_labels = loadData(TRAIN_LINK)
    test_datas, test_labels = loadData(TEST_LINK)

    print (train_labels)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        for epoch in range(2000):
            sess.run(train_step, feed_dict={x:train_datas, y:train_labels, keep_prob:0.7})
            train_accuracy = sess.run(accuracy, feed_dict ={x:train_datas, y:train_labels, keep_prob:DROP_OUT})
            test_accuracy = sess.run(accuracy, feed_dict ={x:test_datas, y:test_labels, keep_prob:DROP_OUT})
            print ("[%d] train_accuracy : %f; test_accuracy : %f"%(epoch, train_accuracy, test_accuracy))
        
        saver.save(sess, '../num_model/num_model.ckpt')
    pass

if __name__ == '__main__':
    main()
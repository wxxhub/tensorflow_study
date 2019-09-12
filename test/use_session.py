import tensorflow as tf
import numpy as np
from PIL import Image

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_LEN = IMAGE_WIDTH * IMAGE_HEIGHT
OUTPUT = 10
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

test_datas, test_labels = loadData(TEST_LINK)

saver = tf.compat.v1.train.import_meta_graph('../num_model/num_model-2000.meta')
graph = tf.compat.v1.get_default_graph()
with tf.Session() as session:
        
        saver.restore(session, tf.train.latest_checkpoint('../num_model'))

        inputs = graph.get_tensor_by_name('inputs:0')
        prediction = graph.get_tensor_by_name('prediction:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')

        pred = session.run(prediction, feed_dict={inputs:test_datas, keep_prob:1})
        print (pred)

"""
Note:2018.3.30
"""

import tensorflow as tf
from tensorflow.python.ops import random_ops
import math
import numpy as np
import glob
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #忽略烦人的警告
BATCH_SIZE = 10
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_CHANNEL = 3

#———————————————————————————————————————图像预处理————————————————————————————————————————————


#从文件队列中读取batch_size个文件，用于训练或测试
def read_tfrecord(serialized, batch_size):

    #parse_single_example解析器将中的example协议内存块解析为张量，
    #每个tfrecord中有多幅图片，但parse_single_example只提取单个样本，
    #parse_single_example只是解析tfrecord，并不对图像进行解码
    features = tf.parse_single_example(
        serialized,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string),
        })

    #将图像文件解码为uint8，因为所有通道的信息都处于0~255，然后reshape
    record_image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(record_image, [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])
    #将label平化为字符串
    label = tf.cast(features['label'], tf.string)

    #用于生成batch的缓冲队列的大小，下面采用的是经验公式
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size

    #生成image_batch和label_batch
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)

    return image_batch, label_batch


# Converting the images to a float of [0,1) to match the expected input to convolution2d
def convert_image(image_batch):
    return (tf.image.convert_image_dtype(image_batch, tf.float32))


# Match every label from label_batch and return the index where they exist in the list of classes
def find_index_label(label_batch):
    return (tf.map_fn(lambda l: tf.where(tf.equal(labels_all, l))[0, 0:1][0],
        label_batch,
        dtype=tf.int64))


#————————————————————————————————————————创建CNN————————————————————————————————————————————————

#占位符，None代表输入的数据个数不确定
image_holder = tf.placeholder(tf.float32,
                              [BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])
label_holder = tf.placeholder(tf.int64, [BATCH_SIZE])
keep_prob_holder = tf.placeholder(tf.float32)  #dropout保留的比例


#此部分代码是创建卷积层时weights_initializer用到的初始化函数，
#书中代码没有此部分，是新添加的
def weights_initializer_random_normal(shape,
                                      dtype=tf.float32,
                                      partition_info=None):
    return random_ops.random_normal(shape)


#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#第1层卷积————————————————————————
with tf.name_scope("conv1") as scope:
    #这里用的是高级层，而不是标准层tf.nn.conv2d，二者的区别见书本第5.3.5节
    conv2d_layer_one = tf.contrib.layers.convolution2d(
        image_holder,
        #产生滤波器的数量，书中代码有误
        num_outputs=32,
        #num_output_channels=32,
        #核尺寸
        kernel_size=(5, 5),
        #激活函数
        activation_fn=tf.nn.relu,
        #权值初始化，书中代码有误：
        #1、weight_init应该是weights_initializer；
        #2、写成tf.random_normal会报错：random_normal() got an unexpected keyword argument 'partition_info'，
        weights_initializer=weights_initializer_random_normal,
        #    weight_init=tf.random_normal,
        stride=(2, 2),
        trainable=True)

#第1层池化————————————————————————————————
with tf.name_scope("pool1") as scope:
    pool_layer_one = tf.nn.max_pool(
        conv2d_layer_one,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')

#第2层卷积————————————————————————————————
with tf.name_scope("conv2") as scope:
    conv2d_layer_two = tf.contrib.layers.convolution2d(
        pool_layer_one,
        #修改，原因同第1层
        num_outputs=64,
        #num_output_channels=64,
        kernel_size=(5, 5),
        activation_fn=tf.nn.relu,
        #修改，原因同第1层
        weights_initializer=weights_initializer_random_normal,
        #weight_init=tf.random_normal,
        stride=(1, 1),
        trainable=True)

#第2层池化————————————————————————————————
with tf.name_scope("pool2") as scope:
    pool_layer_two = tf.nn.max_pool(
        conv2d_layer_two,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME')

#展开层，展开为秩1张量——————————————————————
with tf.name_scope("flat") as scope:
    flattened_layer_two = tf.reshape(pool_layer_two, [BATCH_SIZE, -1])

#全连接层1—————————————————————————————————
with tf.name_scope("full_connect1") as scope:
    hidden_layer_three = tf.contrib.layers.fully_connected(
            flattened_layer_two,
            1024,
            #修改，原因同第1层
            weights_initializer=lambda i, dtype, partition_info=None: tf.truncated_normal([65536, 1024], stddev=0.1),
            #weight_init=lambda i, dtype: tf.truncated_normal([38912, 512], stddev=0.1),
            activation_fn=tf.nn.relu)
    #小trick：dropout
    hidden_layer_three = tf.nn.dropout(hidden_layer_three, keep_prob_holder)

#全连接层2—————————————————————————————————
with tf.name_scope("full_connect2") as scope:
    final_fully_connected = tf.contrib.layers.fully_connected(
            hidden_layer_three,
            120,
            #修改，原因同第1层
            weights_initializer=lambda i, dtype, partition_info=None: tf.truncated_normal([1024, 120], stddev=0.1)
            #weight_init=lambda i, dtype: tf.truncated_normal([512, 120], stddev=0.1)
            )

#输出———————————————————————
with tf.name_scope("output") as scope:
    logits = final_fully_connected
    #查找排名第1的分类结果是否是实际的种类
    top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

#————————————————————————————————————————loss————————————————————————————————————————————————
#计算交叉熵
def loss(logits, labels):
    #按照tensorflow1.0以上版本修改
    #logits是全连接层的输出，不需softmax归一化，因为sparse_softmax_cross_entropy_with_logits函数会先将logits进行softmax归一化，然后与label表示的onehot向量比较，计算交叉熵。
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))


#————————————————————————————————————————training———————————————————————————————————————————————
#模型训练
def training(loss_value, learning_rate, batch):
    return tf.train.AdamOptimizer(learning_rate, 0.9).minimize(
        loss_value, global_step=batch)


#————————————————————————————————————————主函数——————————————————————————————————————————————————

if __name__ == '__main__':

    #下面的几句是我添加的，因为我这里读到的路径形式为：'./imagenet-dogs\\n02085620-Chihuahua\\'，路径分隔符中除第1个之外，都是2个反斜杠，与例程不一致。这里将2个反斜杠替换为斜杠。
    #glob.glob 用于获取所有匹配的路径
    glob_path = glob.glob(r"G:\AI\Images\*")
    #读取所有的label，形式为n02085620-Chihuahua....
    labels_all = list(map(lambda c: c.split("\\")[-1], glob_path))

    #将所有的文件名列表（由函数tf.train.match_filenames_once匹配产生）
    #生成一个队列，供后面的文件阅读器reader读取
    #训练数据队列
    filename_queue_train = tf.train.string_input_producer(
        tf.train.match_filenames_once("F:/TS/TS_p_c/output/training-images/*.tfrecords"))
    #测试数据队列
    filename_queue_test = tf.train.string_input_producer(
        tf.train.match_filenames_once("F:/TS/TS_p_c/output/testing-images/*.tfrecords"))

    #创建tfrecord阅读器，并读取数据。
    #默认shuffle=True，将文件打乱
    reader = tf.TFRecordReader()

    _, serialized_train = reader.read(filename_queue_train)
    _, serialized_test = reader.read(filename_queue_test)

    #读取训练数据——————————————————————————————————
    train_image_batch, train_label_batch = read_tfrecord(
        serialized_train, BATCH_SIZE)
    # Converting the images to a float of [0,1) to match the expected input to convolution2d
    train_images_op = convert_image(train_image_batch)
    # Match every label from label_batch and return the index where they exist in the list of classes
    train_labels_op = find_index_label(train_label_batch)

    #读取测试数据——————————————————————————————————
    test_image_batch, test_label_batch = read_tfrecord(serialized_test,
                                                       BATCH_SIZE)
    # Converting the images to a float of [0,1) to match the expected input to convolution2d
    test_images_op = convert_image(test_image_batch)
    # Match every label from label_batch and return the index where they exist in the list of classes
    test_labels_op = find_index_label(test_label_batch)
    #————————————————————————————————————————————
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.01, batch * 3, 120, 0.95, staircase=True)

    loss_op = loss(logits, train_labels_op)
    train_op = training(loss_op, learning_rate, batch)

    sess = tf.InteractiveSession()
    #必须同时有全局变量和局部变量的初始化，不然会报错：
    #OutOfRangeError (see above for traceback): RandomShuffleQueue '_134_shuffle_batch_8/random_shuffle_queue' is closed and has insufficient elements (requested 3, current size 0)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    #声明一个Coordinator类来协同多个线程
    coord = tf.train.Coordinator()
    # 开始 Queue Runners (队列运行器)
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #执行训练————————————————————————————————————————————
    for j in range(100):
        train_images = sess.run(train_images_op)
        train_labels = sess.run(train_labels_op)
        #print(sess.run(train_label_batch))
        #print(train_labels)
        train_logits, train_result, _ = sess.run(
            [logits, top_k_op, train_op],
            feed_dict={
                image_holder: train_images,
                label_holder: train_labels,
                keep_prob_holder: 0.5
            })
        #print(train_logits)
        #print(train_result)
        if j % 10 == 0:
            #            print(train_labels)
            #            print(train_result)
            print("loss = ",
                  sess.run(
                      loss_op,
                      feed_dict={
                          image_holder: train_images,
                          label_holder: train_labels,
                          keep_prob_holder: 1
                      }), 't=', j)

    #测试————————————————————————————————————————————
    #测试轮数
    num_examples = 1000
    num_iter = int(math.ceil(num_examples/BATCH_SIZE))
    total_sample_count = num_iter*BATCH_SIZE
    true_count = 0
    #测试总准确度
    accuracy_total = 0
    step = 0
    while step < num_iter:
        test_images = sess.run(test_images_op)
        test_labels = sess.run(test_labels_op)
        prediction = sess.run(
            top_k_op,
            feed_dict={
                image_holder: test_images,
                label_holder: test_labels,
                keep_prob_holder: 1.0
            })
        true_count += np.sum(prediction)
        step += 1
        tem_prediction = true_count/(step*BATCH_SIZE)
        if step % 10 == 0:
            print("第", step, "轮测试，准确率为：%.3f, 其中top_1为： %d" % (tem_prediction, np.sum(prediction)))

    predictions = true_count/total_sample_count
    print("总准确率为：%.3f" % predictions)

    #        if i%10 == 0:
    #            print("次数：",i,"————————————————————————————————")
    #            print(test_labels)
    #            print(test_result)

    #结束————————————————————————————————————————————
    #通知其他线程退出
    coord.request_stop()
    #等待所有线程退出
    coord.join(threads)
    sess.close()
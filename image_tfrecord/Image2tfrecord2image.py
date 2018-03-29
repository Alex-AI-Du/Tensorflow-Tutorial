import os
import tensorflow as tf
from PIL import Image
import glob
from itertools import groupby
from collections import defaultdict
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #忽略烦人的警告
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_CHANNEL = 3
def split_image_dataset(path, training_dataset, testing_dataset = False):
    image_filenames = glob.glob(path)
    image_filename_with_breed = list(map(lambda filename: (filename.split("\\")[-2], filename), image_filenames))
    for category, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
        for i, breed_image in enumerate(breed_images):
            if i % 5 == 0 and testing_dataset != False:
                testing_dataset[category].append(breed_image[1])
            else:
                training_dataset[category].append(breed_image[1])
        if testing_dataset != False:
            category_training_count = len(training_dataset[category])
            category_testing_count = len(testing_dataset[category])
            category_training_count_float = float(category_training_count)
            category_testing_count_float = float(category_testing_count)
            assert round(category_testing_count_float / (category_training_count_float + category_testing_count_float), 2) > 0.18, "Not enough testing images."
    if testing_dataset != False:
        print("training_dataset testing_dataset END ------------------------------------------------------")
    else:
        print("training_dataset END ------------------------------------------------------")

# 制作TFRecord文件
def makeTFRecord(dataset, record_location, sess, tfread = False, rows=IMAGE_WIDTH, cols=IMAGE_HEIGHT):
    if not os.path.exists(record_location):
        print("目录 %s 不存在，自动创建中..." % (record_location))
        os.makedirs(record_location)
    writer = None
    current_index = 0
    for category, images_filenames in dataset.items():
        for image_filename in images_filenames:
            if current_index % 100 == 0:
                if writer:
                    writer.close()
                record_filename = "{record_location}-{current_index}.tfrecords".format(
                    record_location=record_location,
                    current_index=current_index)
                writer = tf.python_io.TFRecordWriter(record_filename)
                print(record_filename + "------------------------------------------------------")
            current_index += 1
            if tfread == True :
                image_file = tf.read_file(image_filename)
                try:
                    image = tf.image.decode_jpeg(image_file)
                except:
                    print(sys._getframe().f_lineno,image_filename)
                    continue
                grayscale_image = tf.image.rgb_to_grayscale(image)
                resized_image = tf.image.resize_images(grayscale_image, [rows, cols])
                image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
            else:
                image = Image.open(image_filename)
                image = image.resize((rows, cols))
                image_bytes = image.tobytes() # 将图片转成二进制
            image_label = category.encode("utf-8")
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
                }))
            writer.write(example.SerializeToString())
    writer.close()
    print("write_records_file testing_dataset training_dataset END------------------------------------------------------")

# 将二进制文件读入图中; rows=makeTFRecord.cols, cols=makeTFRecord.rows
def read_and_decode(filequeuelist, rows=IMAGE_HEIGHT, cols=IMAGE_WIDTH):
    fileName_Queue = tf.train.string_input_producer(tf.train.match_filenames_once(filequeuelist))  # 生成一个文件队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(fileName_Queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                     features={
                                        'label': tf.FixedLenFeature([], tf.string),
                                        'image': tf.FixedLenFeature([], tf.string)
                                     })  # 将image数据个label提取出来
    img = tf.decode_raw(features['image'], tf.uint8)
    img = tf.reshape(img, [rows, cols, IMAGE_CHANNEL])  # 将图片的reshape为128*128的3通道图片
    #img = tf.cast(img, tf.float32) * (1.0 / 255) - 0.5  # 在流中抛出img张量
    label = tf.cast(features['label'], tf.string)
    return img, label

def dispaly_image(filequeuelist, save_dir, sess):
    # 创建文件存放目录
    if not os.path.exists(save_dir):
        print("目录 %s 不存在，自动创建中..." % (save_dir))
        os.makedirs(save_dir)
    # 生成每个文件的路径
    fileName_Queue = tf.train.string_input_producer(tf.train.match_filenames_once(filequeuelist))# 生成一个文件队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(fileName_Queue)# 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                        features={
                           'label': tf.FixedLenFeature([], tf.int64),
                           'image': tf.FixedLenFeature([], tf.string)
                        })# 取出包含image和label的feature对象
    img = tf.decode_raw(features['image'], tf.uint8)
    img = tf.reshape(img, [128, 128, 3]) # 将图片的reshape为128*128的3通道图片
    label = tf.cast(features['label'], tf.int32)
    #print(img.shape)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)

    #print("----------",sess.run(img.shape),sess.run(label.shape))
    for i in range(5):
        example, l = sess.run([img, label])  # 在会话中取出image和label
        #print (example.shape, l.shape)
        # 变量名同名的话要注意
        #img = Image.fromarray(exaple, 'RBG')
        image=Image.fromarray(example, 'RGB')#这里Image是之前提到的
        path = r"%s\%s\%s.jpg" % (save_dir,l,i)
        image.save(path)#存下图片
        #print(example, l)
    coord.request_stop()
    coord.join(threads)

if __name__ == "__main__":
    sess = tf.InteractiveSession()
    training_dataset = defaultdict(list)
    testing_dataset = defaultdict(list)
    # 图片分类存放的源路径
    cwd = r"G:\AI\Images\n02085620-Chihuahua\*.jpg"
    # 解析tfrecord文件后图片存放的路径
    save_dir = r"G:\AI\test"
    #split_image_dataset(cwd, training_dataset, testing_dataset)
    # 将图片转成tfrecord文件(图片会变成120*120*3的格式)
    makeTFRecord(training_dataset, "F:/TS/TS_p_c/test/training-images/training-image", sess)
    makeTFRecord(testing_dataset, "F:/TS/TS_p_c/test/testing-images/testing-image", sess)
    # 将tfrecord文件转成图片(图片会变成120*120*3的格式)可以自己在源码中修改图片大小
    [img, label] = read_and_decode("F:/TS/TS_p_c/output/training-images/training-image-0.tfrecords")
    #img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=18, capacity=2000, min_after_dequeue=100,num_threads=2)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # print("----------",sess.run(img.shape),sess.run(label.shape))
    for i in range(1000):
        example, l = sess.run([img, label])  # 在会话中取出image和label
        l = l.decode()
        #print(l)
        # 变量名同名的话要注意
        image = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
        path = r"%s\%s\%s.jpg" % (save_dir, l, i)
        if not os.path.exists(r"%s\%s" % (save_dir, l)):
            print("目录 %s\%s 不存在，自动创建中..." % (save_dir, l))
            os.makedirs(r"%s\%s" % (save_dir, l))
        image.save(path)  # 存下图片
        # print(example, l)
    coord.request_stop()
    coord.join(threads)
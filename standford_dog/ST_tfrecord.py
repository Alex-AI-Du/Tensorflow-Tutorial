"""
Note:2018.3.30
"""

import tensorflow as tf
import glob
from itertools import groupby
from collections import defaultdict
from PIL import Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #忽略烦人的警告
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

sess = tf.InteractiveSession()

#查找符合一定规则的所有文件，并将文件名以lis形式返回。
#image_filenames = glob.glob(r"G:\AI\Images\n02110*\*.jpg")
image_filenames = glob.glob(r"G:\AI\Images\n02*\*.jpg")

#这句是我添加的。因为读到的路径形式为：'./imagenet-dogs\\n02085620-Chihuahua\\n02085620_10074.jpg'，路径分隔符中除第1个之外，都是2个反斜杠，与例程不一致。这里将2个反斜杠替换为斜杠
#image_filenames = list(map(lambda image: image.replace('\\', '/'), image_filenames_0))

#用list类型初始化training和testing数据集，用defaultdict的好处是为字典中不存在的键提供默认值
training_dataset = defaultdict(list)
testing_dataset = defaultdict(list)

#将品种名从文件名中切分出，image_filename_with_breed是一个迭代器，用list(image_filename_with_breed)将其转换为list，其中的元素类似于：('n02085620-Chihuahua', './imagenet-dogs/n02085620-Chihuahua/n02085620_10131.jpg')。
image_filename_with_breed = list(map(lambda filename: (filename.split("\\")[-2], filename), image_filenames))

## Group each image by the breed which is the 0th element in the tuple returned above
#groupby后得到的是一个迭代器，每个元素的形式为：('n02085620-Chihuahua', <itertools._grouper at 0xd5892e8>)，其中第1个元素为种类；第2个元素代表该类的文件，这两个元素也分别对应for循环里的dog_breed和breed_images。
for dog_breed, breed_images in groupby(image_filename_with_breed,
                                       lambda x: x[0]):

    #enumerate的作用是列举breed_images中的所有元素，可同时返回索引和元素，i和breed_image
    #的最后一个值分别是：168、('n02116738-African_hunting_dog', './imagenet-dogs/
    #n02116738-African_hunting_dog/n02116738_9924.jpg')
    for i, breed_image in enumerate(breed_images):

        #因为breed_images是按类分别存储的，所以下面是将大约20%的数据作为测试集，大约80%的
        #数据作为训练集。
        #testing_dataset和training_dataset是两个字典，testing_dataset中
        #的第一个元素是 'n02085620-Chihuahua': ['./imagenet-dogs/n02085620-Chihuahua/
        #n02085620_10074.jpg', './imagenet-dogs/n02085620-Chihuahua/
        #n02085620_11140.jpg',.....]
        if i % 5 == 0:
            testing_dataset[dog_breed].append(breed_image[1])
        else:
            training_dataset[dog_breed].append(breed_image[1])

    # 测试每种类型下的测试集是否至少包含了18%的数据
    breed_training_count = len(training_dataset[dog_breed])
    breed_testing_count = len(testing_dataset[dog_breed])

    assert round(breed_testing_count /
                 (breed_training_count + breed_testing_count),
                 2) > 0.18, "Not enough testing images."


def write_records_file(dataset, record_location):
    """
    Fill a TFRecords file with the images found in `dataset` and include their category.

    Parameters
    ----------
    dataset : dict(list)
      Dictionary with each key being a label for the list of image filenames of its value.
    record_location : str
      Location to store the TFRecord output.
    """
    if not os.path.exists(record_location):
        print("目录 %s 不存在，自动创建中..." % (record_location))
        os.makedirs(record_location)
    writer = None

    # Enumerating the dataset because the current index is used to breakup the files if they get over 100
    # images to avoid a slowdown in writing.
    current_index = 0
    #遍历每一种类型的所有文件
    for breed, images_filenames in dataset.items():
        #遍历每一个文件
        for image_filename in images_filenames:
            if current_index % 1000 == 0:
                if writer:
                    writer.close()
                #创建tensorflow record的文件名
                record_filename = "{record_location}-{current_index}.tfrecords".format(
                    record_location=record_location,
                    current_index=current_index)

                writer = tf.python_io.TFRecordWriter(record_filename)
            current_index += 1

            '''
            image_file = tf.read_file(image_filename)

            #将图片按照jpeg格式解析，ImageNet dogs中有些图片按照JPEG解析时会出错，用try
            #语句忽视解析错误的图片。
            try:
                image = tf.image.decode_jpeg(image_file)
            except:
                print(image_filename)
                continue

            # 转换为灰度图像.经测试最好不要转换灰度，grayscale_image会是增加原图像的10倍处理时间。绝对是个坑！！！
            #grayscale_image = tf.image.rgb_to_grayscale(image)
            #此处做了修改，resize_images的第二个参数要求是tensor，原代码有误。
            #resized_image = tf.image.resize_images(grayscale_image, 250, 151)
            resized_image = tf.image.resize_images(image, [250, 151])

            # tf.cast is used here because the resized images are floats but haven't been converted into
            # image floats where an RGB value is between [0,1).
            image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
            '''
            #使用Image.open读取图像比tf.read_file的速度快10倍，建议使用Image.open
            image = Image.open(image_filename)
            image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            image_bytes = image.tobytes()  # 将图片转成二进制
            # Instead of using the label as a string, it'd be more efficient to turn it into either an
            # integer index or a one-hot encoded rank one tensor.
            # https://en.wikipedia.org/wiki/One-hot
            #将表示种类的字符串转换为python默认的utf-8格式，防止有问题
            image_label = breed.encode("utf-8")

            ## 创建一个 example protocol buffer 。
            # 其中，feature={
            # 'label':
            # tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
            # 'image':
            # tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            # })是创建1个属性
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'label':
                    tf.train.Feature(bytes_list=tf.train.BytesList(
                        value=[image_label])),
                    'image':
                    tf.train.Feature(bytes_list=tf.train.BytesList(
                        value=[image_bytes]))
                }))
            #SerializeToString()将文件序列化为二进制字符串
            writer.write(example.SerializeToString())
    writer.close()

#分别将测试数据和训练数据写入tensorflow record，分别保存在文件夹./output/testing-images/和./output/
#training-images/下面。
write_records_file(testing_dataset, "F:/TS/TS_p_c/output/testing-images/testing-image")
write_records_file(training_dataset, "F:/TS/TS_p_c/output/training-images/training-image")
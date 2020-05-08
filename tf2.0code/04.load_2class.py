# -*- coding: utf-8 -*-
"""
Created on Thu May  7 20:20:48 2020

@author: lihuanyu
"""

import tensorflow as tf
print('Tensorflow version: {}'.format(tf.__version__))
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

data_dir = r"C:\Users\lihuanyu\Desktop\日月光华Tensorflow\日月光华-tensorflow资料\数据集\2_class"
data_dir = pathlib.Path(data_dir)
#目录的数量
image_count = len(list(data_dir.glob('*/*.jpg')))
#显示类别
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
print(CLASS_NAMES)
# 打印该路径下的文件
for item in data_dir.iterdir():
    print(item)
import random  
all_image_path = list(data_dir.glob("*/*"))
all_image_path = [str(path) for path in all_image_path]
random.shuffle(all_image_path)

image_count = len(all_image_path)
print(image_count)
print(all_image_path[:10])
import IPython.display as display
for n in range(3):
  image_path = random.choice(all_image_path)
  display.display(display.Image(image_path))
#确定每个图像的标签
lable_names = sorted(item.name for item in data_dir.glob("*/"))
#为每个标签分配索引,构建字典
lable_to_index = dict((name,index) for index,name in enumerate(lable_names))
print(lable_to_index)
#创建一个列表，包含每个文件的标签索引
all_image_label = [lable_to_index[pathlib.Path(path).parent.name] for path in all_image_path]
#%%加载和格式化图片
#这里加载的是原始数据
img_raw = tf.io.read_file(all_image_path[0])
#将它解码为张量
img_tensor = tf.image.decode_image(img_raw)
print(img_tensor.shape,img_tensor.dtype)
#我们可以根据需要调整模型的大小
img_final = tf.image.resize(img_tensor,[192,192])
img_final = img_final/255.0
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())

#包装为函数，以备后用
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image /= 255.0  # normalize to [0,1] range
    return image
#加载图片
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


#%%
image_path = all_image_path[0]
label = all_image_label[0]

plt.imshow(load_and_preprocess_image(image_path))
plt.grid(False)
##plt.xlabel(caption_image(image_path))
plt.title(lable_names[label].title())
print()
#%%构建一个tf.data.Dataset
#一个图片数据集构建 tf.data.Dataset 最简单的方法就是使用 from_tensor_slices 方法。
#将字符串数组切片，得到一个字符串数据集：
path_ds =  tf.data.Dataset.from_tensor_slices(all_image_path)
print(path_ds)
#现在创建一个新的数据集，通过在路径数据集上映射 preprocess_image 来动态加载和格式化图片。
AUTOTUNE = tf.data.experimental.AUTOTUNE
image_ds = path_ds.map(load_and_preprocess_image,num_parallel_calls=AUTOTUNE)

plt.figure(figsize=(8,8))
for n, image in enumerate(image_ds.take(4)):
  plt.subplot(2,2,n+1)
  plt.imshow(image)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  #plt.xlabel(caption_image(all_image_path[n]))
  plt.show()

#%%一个（图片，标签）对数据集
lable_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_label,tf.int64))
for label in lable_ds.take(5):
    #print(label)
    print(lable_names[label.numpy()])
#因为这些数据集顺序相同，可以将他们打包起来
image_label_ds = tf.data.Dataset.zip((image_ds,lable_ds))
print(image_label_ds)
#注意：当你拥有形似 all_image_labels 和 all_image_paths 的数组，tf.data.dataset.Dataset.zip 的替代方法是将这对数组切片
# =================================im============================================
# ds = tf.data.Dataset.from_tensor_slices((all_image_path,all_image_label))
# def load_and_preprocess_from_path_label(path, label):  
#     return load_and_preprocess_image(path),label
# image_label_ds = ds.map(load_and_preprocess_from_path_label)    
# =============================================================================
#%%设置训练数据和测试数据的大小

test_count = int(image_count*0.2)
train_count = image_count - test_count
print(test_count,train_count)
#跳过test_count个
train_dataset = image_label_ds.skip(test_count)
test_dataset = image_label_ds.take(test_count)
#%%开始训练
'''
训练我们会将数据
1.充分的打乱
2.被分割batch'
3.永远的重复
4.尽快的batch
'''
batch_size = 32
# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据被充分打乱。
train_ds = train_dataset.shuffle(buffer_size=image_count).repeat().batch(batch_size)
test_ds = test_dataset.batch(batch_size)
# # 当模型在训练的时候，`prefetch` 使数据集在后台取得 batch。
# ds = ds.prefetch(buffer_size=AUTOTUNE)
'''
我们需要注意的是
1.我们在.repeat在进行.shuffl会出现一种情况，那就是会在epoh之间打乱数据，当有的数据出现两次的时候，有的数据还没有被打乱
2.在batch之后shuffle,会打乱 batch 的顺序，但是不会在 batch 之间打乱数据。
3.你在完全打乱中使用和数据集大小一样的 buffer_size（缓冲区大小）。较大的缓冲区大小提供更好的随机化，但使用更多的内存，直到超过数据集大小。
4.在从随机缓冲区中拉取任何元素前，要先填满它。所以当你的 Dataset（数据集）启动的时候一个大的 buffer_size（缓冲区大小）可能会引起延迟。
5.在随机缓冲区完全为空之前，被打乱的数据集不会报告数据集的结尾。Dataset（数据集）由 .repeat 重新启动，导致需要再次等待随机缓冲区被填满。
'''
#%%数据标准化
model = tf.keras.Sequential()   #顺序模型
model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(1024, (3, 3), activation='relu'))
model.add(tf.keras.layers.GlobalAveragePooling2D())

model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#%%
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)
#%%
steps_per_eooch = train_count//batch_size
validation_steps = test_count//batch_size

history = model.fit(train_ds, epochs=30, steps_per_epoch=steps_per_eooch, validation_data=test_ds, validation_steps=validation_steps)
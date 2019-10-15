'''
author ambition
date 
function 从图片数据集而非自带keras自带数据包中导入数据
因为Keras的Sequential类只支持顺序模型，所以可以使用以返回值形式定义网络结构
version
'''
import keras
from keras.layers import Dense,Conv2D,Flatten,Input,MaxPooling2D
import tensorflow as tf
from keras.models import Sequential
from keras import optimizers
from keras import backend
from keras.models import Model
import os
import cv2 as cv
import numpy as np

classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
BATCH_SIZE=64
train_path='./cifar-10/train'
test_path='./cifar-10/test'
MODEL_SAVE_PATH="./model/"#保存训练好的模型
MODEL_NAME="AlexNet_model"


def dataset(path):
    # writer = tf.python_io.TFRecordWriter("train.tfrecords")
    data_sets=[]
    data_labels=[]
    k=0
    for index, name in enumerate(classes):
        class_path = path +'/'+ name + '/'
        for i,image_name in enumerate(os.listdir(class_path)):
            image_path=class_path+image_name
            img=cv.imread(image_path)
            k+=1
            # print(img.shape)
            # print(type(img))
            data_sets.append(img)
            #整理成标签
            onehot = []
            hot=classes.index(name)
            for j in range(10):
                if j ==hot:
                    onehot.append(1)
                else:
                    onehot.append(0)
            # print(onehot)
            data_labels.append(onehot)

    data_sets=np.reshape(data_sets,[k,32,32,3])
    data_labels=np.reshape(data_labels,[k,10])
    # 图象处理标准化
    data_sets = data_sets.astype('float')
    data_sets /= 255
    return data_sets,data_labels


global_step=tf.Variable(0,trainable=False)
x_train,y_train=dataset(train_path)
x_test,y_test=dataset(test_path)

input=Input(dtype=tf.float32,shape=(32,32,3))
#第一层
layer1=Conv2D(64,(5,5),strides=1,padding='valid',activation='relu')(input)
#第二层
layer2=MaxPooling2D(pool_size=(2,2),strides=(2,2))(layer1)
#第三层
layer3=Conv2D(64,(5,5),strides=1,padding='valid',activation='relu')(layer2)
#第四层
layer4=MaxPooling2D([3,3],strides=2)(layer3)
#进入全连接层
nodes=Flatten()(layer4)
layer5=Dense(384,activation='relu')(nodes)
layer6=Dense(192,activation='relu')(layer5)
predictions=Dense(10,activation='softmax')(layer6)

model=Model(inputs=input,outputs=predictions)
# model.summary()
# plot_model(model, to_file='keras_alexnet.png')
# SVG(model_to_dot(model).create(prog='dot', format='svg'))
#配置训练过程
sgd =optimizers.SGD(lr=0.01, momentum=0.9)
#优化方法
model.compile(optimizer=sgd, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])#损失函数,metrics评价函数
#
# sgd = optimizers.SGD(lr=0.01, momentum=0.9)
# model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=25, batch_size=64,validation_data=(x_test,y_test))
#评价函数
train_eva=model.evaluate(x_train, y_train, batch_size=64)
test_eva=model.evaluate(x_test, y_test, batch_size=64)
scroe=model.evaluate(x_test,y_test,)
# saver=tf.train.Saver()
# with tf.Session() as sess:
#     init=tf.global_variables_initializer()
#     sess.run((init))
#     ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
#     if ckpt and ckpt.model_checkpoint_path:
#         saver.restore(sess, ckpt.model_checkpoint_path)
#     saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

print("test loss:",scroe[0])
print("test accuracy:",scroe[1])


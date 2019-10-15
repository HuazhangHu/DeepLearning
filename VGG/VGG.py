'''
author ambition
date 
function 基于keras的VGG经典神经网络的实现
version
'''

import keras
from keras.layers import Dense,Conv2D,Flatten,Input,MaxPooling2D
import tensorflow as tf
from keras.models import Sequential
from keras import optimizers
from keras import backend
from keras.models import Model,load_model,model_from_json
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

def conv(inputs,ksize,num_filters,strides,padding,scope='convs'):
    '''
    自己定义的卷积神经网络
    :param inputs: 输入
    :param ksize: 卷积核大小
    :param num_filters: 卷积核个数
    :param strides: 步长
    :param padding: 0填充
    :param scope: 变量域
    :param reuse: 复用
    :return:
    '''
    conv=Conv2D(filters=num_filters,kernel_size=ksize,strides=strides,padding=padding)(inputs)
    return conv

def vgg_block(inputs,num_convs,out_depth,num_pool,scope='vgg_block',reuse=None):
    '''
    :param inputs: 输入
    :param num_convs: 卷积层的个数
    :param out_depth: 每一层卷积核的个数
    :param scope: 变量域名
    :param reuse: 是否复用
    :return: VGG-blocK
    '''
    with tf.variable_scope(scope,reuse=reuse):
        net=inputs
        for i in range(num_convs):
            net=Conv2D(filters=out_depth,kernel_size=[3,3],strides=[1,1],padding='SAME')(net)
        net=MaxPooling2D([2,2],strides=[2,2],name='pool%d'%num_pool)(net)

        return net

def vgg_stack(inputs,num_convs,out_depths,scope='vgg_stack',reuse=None):
    '''
    搭建VGG-stack
    :param inputs: 输入
    :param num_convs: 每一个VGG-block里卷积层的个数
    :param out_depths: 各一个block里卷积核的个数
    :param scope:
    :param reuse:
    :return:
    '''
    num_pool=1
    with tf.variable_scope(scope,reuse=reuse) as sc:
        net=inputs
        for i,(n,d) in enumerate(zip(num_convs,out_depths)):
            net=vgg_block(net,n,d,num_pool,scope='block%d'%i,reuse=None)
            num_pool+=1
        return net

def vgg(inputs,num_convs,out_depths,num_output,scope='vgg',reuse=None):
    '''
    构建VGG网络
    :param inputs:
    :param num_convs: 每一个VGG-block里卷积层的个数 [...,...,...,...]
    :param out_depths: 各一个block里卷积核的个数 [...,...,...,...]
    :param scope:
    :param reuse:
    :return:
    '''
    with tf.variable_scope(scope,reuse=reuse) as sc:
        net=vgg_stack(inputs,num_convs,out_depths=out_depths)
        with tf.variable_scope("classification"):
            net=Flatten()(net)#
            # net=tf.reshape(net,(64,-1))
            net=Dense(100,activation='relu')(net)
            predictions = Dense(num_output, activation='softmax')(net)
        return predictions
#
# def keras_
x_train,y_train=dataset(train_path)
x_test,y_test=dataset(test_path)


input=Input(dtype=tf.float32,shape=(32,32,3))
train_predictions=vgg(input,(1,1,2,2,2),(64,128,256,512,512),10,reuse=None)
model=Model(inputs=input,outputs=train_predictions)#前向传播搭建
#构建反向传播
sgd =optimizers.SGD(lr=0.01, momentum=0.9)
#优化方法
model.compile(optimizer=sgd, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])#损失函数,metrics评价函数

model.fit(x=x_train,y=y_train,batch_size=64,epochs=1,validation_data=(x_test,y_test))
# test_predictions=vgg(x_test,(1,1,2,2,2),(64,128,256,512,512),10,reuse=True)
# with tf.variable_scope('loss'):
#     train_loss=tf.losses.sparse_softmax_cross_entropy(labels=y_train,logits=train_predictions,scope='train')
#     # val_loss=tf.losses.sparse_softmax_cross_entropy(labels=y_test,logits=test_predictions,scope='valid')
#     train_step=tf.train.GradientDescentOptimizer(0.01)(train_loss)

train_eva=model.evaluate(x_train, y_train, batch_size=64)
score=model.evaluate(x_test, y_test, batch_size=64)
print("test loss:",score[0])
print("test accuracy:",score[1])

model.save('model.h5')#保存模型
#
# #模型载入
# model=load_model('model.h5')
# score=model.evaluate(x_test, y_test, batch_size=64)
#
# #保存模型的参数
# model.save_weights('model_weights.h5')
# #载入模型的参数
# model.load_weights('model_weights.h5')
# #保存网络结构成json
# json_string=model.to_json()
# #从json载入网络结构
# model=model_from_json(json_string)





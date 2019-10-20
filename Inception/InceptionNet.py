'''
author ambition
date 2019.10.20
function inception网络
version 2.0
'''

import keras
from keras.layers import Dense,Conv2D,Flatten,Input,MaxPooling2D
import tensorflow as tf
from keras.models import Sequential
from keras import optimizers
from keras import backend
from keras.models import Model
from keras.datasets import cifar10
import tensorflow.contrib.slim as slim
import os
import cv2 as cv
import numpy as np

classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
BATCH_SIZE=50
train_path='./cifar-10/train'
test_path='./cifar-10/test'
MODEL_SAVE_PATH="./model/"#保存训练好的模型
MODEL_NAME="InceptionNet_model"

def dataset(path):
    '''
    将x转换为矩阵，Y为类别序号
    :param path:
    :return:
    '''
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

def inception(x,d0_1,d1_1,d1_3,d2_1,d2_5,d3_1,scope='inception',reuse=None):
    '''
    inception模块
    :param x:输出
    :param d0_1:第一分支1x1卷积核的个数
    :param d1_1:第二分支第一层1x1卷积...
    :param d1_3:第二分支第二层3x3卷积
    :param d2_1:第三分支第一层1x1卷积
    :param d2_5:第三分支第二层5x5卷积
    :param d3_1:第四分支第二层1x1卷积核的个数
    :param scope:
    :param reuse:
    :return: 返回一个inception模块
    '''
    with tf.variable_scope(scope,reuse=reuse):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d],stride=1,padding='SAME'):
            #第一个分支:1*1卷积
            with tf.variable_scope('branch0'):
                branch_0=slim.conv2d(x,d0_1,[1,1],scope='conv_1x1')

            #第二个分支：1*1卷积+3*3卷积
            with tf.variable_scope('branch1'):
                branch_1=slim.conv2d(x,d1_1,[1,1],scope='conv_1x1')
                branch_1=slim.conv2d(branch_1,d1_3,[3,3],scope='conv_3x3')

            #第三个分支:1*1卷积+5*5卷积
            with tf.variable_scope('branch2'):
                branch_2=slim.conv2d(x,d2_1,[1,1],scope='conv_1x1')
                branch_2=slim.conv2d(branch_2,d2_5,[5,5],scope='conv_5x5')

            #第四个分支：3*3最大池化+1*1卷积
            with tf.variable_scope('branch3'):
                branch_3=slim.max_pool2d(x,[3,3],scope='max_pool')
                branch_3=slim.conv2d(branch_3,d3_1,[1,1],scope='conv_1x1')

            #将各个分支拼接起来
            net=tf.concat([branch_0,branch_1,branch_2,branch_3],axis=-1)

            return net

def googlenet(inputs,num_class,is_training=None,verbose=False,reuse=None):
    '''
    将inception模块连接成googlenet网络
    :param inputs:
    :param num_class:
    :param reuse:
    :param is_training: 当is_trainning==true时，batch_norm使用的是batch数据的移动平均，方差值是固定值
    :param verbose:
    :return:
    '''
    # 通过arg_scope()给函数设定默认参数
    with tf.variable_scope('googlenet',reuse=reuse):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], padding='SAME', stride=1):
                net = inputs
                with tf.variable_scope('block1'):
                    net = slim.conv2d(net, 64, [5, 5], stride=2, scope='conv_5x5')

                    if verbose:
                        print('block1 output:{}'.format(net.shape))

                with tf.variable_scope('block2'):
                    net = slim.conv2d(net, 64, [1, 1], scope='conv_1x1')
                    net = slim.conv2d(net, 192, [3, 3], scope='conv_3x3')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='max_pool')

                    if verbose:
                        print('block2 output:{}'.format(net.shape))

                with tf.variable_scope('block3'):
                    net = inception(net, 64, 96, 128, 16, 32, 32, scope='inception_1')
                    net = inception(net, 128, 128, 192, 32, 96, 64, scope='inception_2')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='max_pool')

                    if verbose:
                        print('block3 output:{}'.format(net.shape))

                with tf.variable_scope('block4'):
                    net = inception(net, 192, 96, 208, 16, 48, 64, scope='inception_1')
                    net = inception(net, 160, 112, 224, 24, 64, 64, scope='inception_2')
                    net = inception(net, 128, 128, 256, 24, 64, 64, scope='inception_3')
                    net = inception(net, 112, 144, 288, 24, 64, 64, scope='inception_4')
                    net = inception(net, 256, 160, 320, 32, 128, 128, scope='inception_5')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='max_pool')

                    if verbose:
                        print('block4 output:{}'.format(net.shape))

                with tf.variable_scope('block5'):
                    net = inception(net, 256, 160, 320, 32, 128, 128, scope='inception_1')
                    net = inception(net, 384, 182, 384, 48, 128, 128, scope='inception_2')
                    net = slim.avg_pool2d(net, [2, 2], stride=2, scope='average_pool')

                    if verbose:
                        print('block5 output:{}'.format(net.shape))

                with tf.variable_scope('classification'):
                    net = slim.flatten(net)
                    net = slim.fully_connected(net, num_class, activation_fn=None, normalizer_fn=None, scope='logit')

                    if verbose:
                        print('block6 output:{}'.format(net.shape))

                return net


if __name__ == '__main__':
    #真实数据
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, y_train = dataset(train_path)
    x_test, y_test = dataset(test_path)
    #用于前向传播搭建网络的占位输入输出
    #设置函数参数默认值
    with slim.arg_scope([slim.conv2d]) as sc:
        conv_scope = sc
        is_training = tf.placeholder(tf.bool, name='is_trainging')
        # 预测结果
        with slim.arg_scope(conv_scope):
            train_out = googlenet(x_train,10,is_training=is_training,verbose=True)
            # val_out=googlenet(x,10,is_training=is_training,reuse=True)

        with tf.variable_scope('loss'):
            print(type(y_train),type(train_out))
            train_loss=tf.nn.softmax_cross_entropy_with_logits(train_out,y_train)
            # print(train_loss)
            # valid_loss=tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=val_out,scope='valid_loss')

        with tf.name_scope('accuracy'):
            with tf.name_scope('train'):
                #tf.cast(input_tensor,dtype)数据类型转换
                #tf.argmax(input,axis)返回指定轴的最大值的索引，axis<=0表示列,axis>0表示行
                #这的y应该为分类结果的标号

                train_acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(train_out,axis=-1,output_type=tf.int32),y_train),tf.float32))

            # with tf.name_scope('valid'):
            #     valid_acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(val_out,axis=-1,output_type=tf.int32),y_test),tf.float32))

        opt=tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9)
        #使用tf.get_collection获得所有需要更新的op
        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #使用tensorflow的控制流tf.control_dependencies(更新操作)，先执行更新算子，再执行训练
        with tf.control_dependencies(update_ops):
            train_op=opt.minimize(train_loss)

        with tf.Session() as sess:
            init=tf.global_variables_initializer()
            sess.run(init)
            for i in range(50000):#
                _,loss,accuracy=sess.run([train_op,train_loss,train_acc],feed_dict={is_training:True})
                if i%1000==0:
                    print('after {} training steps,the loss is {},the accuray is {}'.format(i,loss,accuracy))




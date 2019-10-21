'''
author ambition
date 
function 
version
'''

import tensorflow as tf
from tensorflow.contrib import slim as slim
import os
import numpy as np
import cv2 as cv


BATCH_SIZE=64
classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
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



def bn_relu_conv(x,out_depth,scope='dense_basic_conv',reuse=None):
    '''
    构建基本卷积单元
    :param x:
    :param out_depth: 输出的深度
    :param scope:
    :param reuse:
    :return:
    '''
    with tf.variable_scope(scope,reuse=reuse):
        net=slim.batch_norm(x,activation_fn=None,scope='bn')
        net=tf.nn.relu(net,name='activation')
        net=slim.conv2d(net,out_depth,3,activation_fn=None,normalizer_fn=None,biases_initializer=None,scope='conv')

        return net

def dense_block(x,growth_rate,num_layers,scope='dense_block',reuse=None):
    '''
    构建dense的基本单元
    :param x:
    :param growth_rate:
    :param num_layers:
    :param scope:
    :param reuse:
    :return:
    '''
    in_depth=x.get_shape().as_list()[-1]

    with tf.variable_scope(scope,reuse=reuse):
        net=x
        for i in range(num_layers):
            out=bn_relu_conv(net,growth_rate,scope='block%d'%i)
            net=tf.concat([net,out],axis=-1)

        return net

def transition(x,out_depth,scope='transition',reuse=None):
    '''
    构建transition模块
    :param x:
    :param out_depth:
    :param scope:
    :param reuse:
    :return:
    '''
    in_depth=x.get_shape().as_list()[-1]

    with tf.variable_scope(scope,reuse=reuse):
        net=slim.batch_norm(x,activation_fn=None,scope='bn')
        net = tf.nn.relu(net, name='activation')
        #1x1卷积核
        net = slim.conv2d(net, out_depth, 1, activation_fn=None, normalizer_fn=None, biases_initializer=None,scope='conv')
        net=slim.avg_pool2d(net,2,2,scope='avg_pool')

        return net
def densenet(x,num_class,growth_rate=32,block_layers=[6,12,24,16],is_training=None,scope='densenet',reuse=None,verbose=False):
    '''
    搭建DenseNet网络结构
    :param x:
    :param num_class:分类数
    :param growth_rate:
    :param block_layers: 每个densenet_block的层数
    :param is_training:训练锁
    :param scope:
    :param reuse:
    :param verbose:
    :return:返回densenet分类的结果
    '''
    with tf.variable_scope(scope,reuse=reuse):
        with slim.arg_scope([slim.batch_norm],is_training=is_training):

            if verbose:
                print('input:{}'.format(x.shape))

            with tf.variable_scope('block1'):
                net=slim.conv2d(x,64,7,2,normalizer_fn=None,activation_fn=None,scope='conv_7x7')
                net=slim.batch_norm(net,activation_fn=None,scope='bn')
                net=tf.nn.relu(net,name='activation')
                net=slim.max_pool2d(net,3,2,scope='max_pool')

                if verbose:
                    print('block1:{}'.format(net.shape))

            for i,num_layers in enumerate(block_layers):
                with tf.variable_scope('block%d'%(i+1)):
                    net=dense_block(net,growth_rate,num_layers)
                    if i !=len(block_layers)-1:
                        current_depth=net.get_shape.as_list()[-1]
                        net=transition(net,current_depth//2)

                if verbose:
                    print('block{}:{}'.format(i+1,net.shape))

            with tf.variable_scope('block%d'%(len(block_layers)+1)):
                net = slim.batch_norm(net, activation_fn=None, scope='bn')
                net = tf.nn.relu(net, name='activation')
                net=tf.reduce_mean(net,[1,2],name='global_pool',keep_dims=True)
                if verbose:
                    print('block{}:{}'.format(len(block_layers)+1,net.shape))

            with tf.variable_scope('classification'):
                net=slim.flatten(net,scope='flatten')
                net=slim.fully_connected(net,num_outputs=num_class,activation_fn=None,normalizer_fn=None,scope='logit')

                if verbose:
                    print('classification{}'.format(net.shape))

                return net


with slim.arg_scope([slim.conv2d],activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm) as sc:
    conv_scope=sc
    x_train, y_train = dataset(train_path)
    x_test, y_test = dataset(test_path)
    is_training = tf.placeholder(tf.bool, name='is_trainning')
    with slim.arg_scope(conv_scope):
        train_out=densenet(x_train,10,is_training=is_training,verbose=True)
        valid_out=densenet(x_test,10,is_training=is_training,verbose=True)

        with tf.variable_scope('loss'):
            #损失函数
            train_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(train_out,y_train,scope='train_loss')
            valid_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(valid_out,y_test,scope='valid_loss')

        with tf.name_scope('accuracy'):
            with tf.name_scope('train'):
                train_acc = tf.reduce_mean(
                    tf.cast(tf.equal(tf.argmax(train_out, axis=-1, output_type=tf.int32), y_train), tf.float32))



with tf.Session() as sess:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(train_loss)
        init = tf.initialize_all_variables()
        sess.run(init)
        for i in range(10000):
            operate, loss_value, acc_value = sess.run([update_ops, train_loss, train_acc],
                                                      feed_dict={is_training: True})

'''
author ambition
date 
function ResNetnetwork基于tensorflow原生封装slim
version
'''

import tensorflow as tf
from keras.datasets import cifar10
from tensorflow.contrib import slim as slim
import numpy as np
import cv2 as cv
import os


BATCH_SIZE=64
# (x_train, y_train), (x_test, y_test) =cifar10.load_data()
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


# def net_batch(inputs,batch_size):
#     net_batch=[]
#     for i,item in enumerate(inputs):
#         net_batch.append(item)
#         if item:
#             raise NameError
#         if i+1%batch_size==0:
#             return net_batch

#首先定义一个下采样函数
def subsample(x,stride,scope=None):
    if stride==1:#如果步长为1，那么大小是不会发生改变
        return x
    #如果步长不为1,例如等于2,就相当于将大小/2
    return slim.max_pool2d(x,[1,1],stride=stride,scope=scope)


def residual_block(x,bottleneck_depth,out_depth,strdie=1,scope='residual_block'):
    '''
    #定义residual_block
    :param x:
    :param bottleneck_depth: 瓶颈层的卷积核个数
    :param out_depth: #输出channel数
    :param strdie:如果stride==1,不管残差块输入输出通道数是否相同，图象大小都不改变,如果不等于,则要改变图象大小(pool or conv)
    :param scope:
    :return:
    '''
    in_depth=x.get_shape().as_list()[-1]

    with tf.variable_scope(scope):
        #如果跳跃相加后的通道数没有改变，则通过下采样（池化）来改变输入x的大小,这样才可以和经过卷积后的y跳跃相加
        if in_depth==out_depth:
            shortcut=subsample(x,strdie,'shortcut')
        #如果通道数发生了改变，则用卷积改变输入的通道(out_depth)以及大小(stride)
        else:
            shortcut=slim.conv2d(x,out_depth,[1,1],stride=strdie,activation_fn=None,scope='shortcut')

        #定义三瓶颈层残差网络
        residual=slim.conv2d(x,bottleneck_depth,[1,1],strdie=1,scope='conv1')
        residual=slim.conv2d(residual,bottleneck_depth,3,strdie,scope='conv2')
        residual=slim.conv2d(residual,out_depth,[1,1],stride=1,activation_fn=None,scope='conv1')
        print(type(residual))
        #相加
        output=tf.nn.relu(shortcut+residual)

        return out_depth


#搭建resNet整体结构
def resnet(inputs,num_class,reuse=None,is_training=None,verbose=False):
    with tf.variable_scope('resnet',reuse=reuse):
        net=inputs
        #打印输入的形状
        if verbose:
            print('input:{}'.format(net.shape))

        with slim.arg_scope([slim.batch_norm],is_training=is_training):
            with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],padding='SAME'):

                with tf.variable_scope('block1'):
                    net=slim.conv2d(net,32,[5,5],stride=2,scope='conv_5x5')
                    if verbose:
                        print('block1:{}'.format(net.shape))

                with tf.variable_scope('block2'):
                    net=slim.max_pool2d(net,[3,3],2,scope='max_pool')
                    net=residual_block(net,32,128,scope='residual_block1')
                    net=residual_block(net,32,128,scope='residual_block2')

                    if verbose:
                        print('block2:{}'.format(net.shape))

                with tf.variable_scope('block3'):
                    net = residual_block(net, 64, 256,strdie=2, scope='residual_block1')
                    net = residual_block(net, 64, 256, scope='residual_block2')

                    if verbose:
                        print('block3:{}'.format(net.shape))

                with tf.variable_scope('block4'):
                    net = residual_block(net, 128, 512,strdie=2, scope='residual_block1')
                    net = residual_block(net, 128, 512, scope='residual_block2')

                    if verbose:
                        print('block4:{}'.format(net.shape))


                with tf.variable_scope('classification'):
                    net=tf.reduce_mean(net,[1,2],name='global_pool',keep_dims=True)
                    net=slim.flatten(net,scope='flatten')
                    net=slim.fully_connected(net,num_class,activation_fn=None,normalizer_fn=None,scope='logit')

                    if verbose:
                        print('block5:{}'.format(net.shape))

                return net

with slim.arg_scope([slim.conv2d],activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm) as sc:
    conv_scope=sc
    #训练锁
    x_train, y_train = dataset(train_path)
    x_test, y_test = dataset(test_path)
    is_training=tf.placeholder(tf.bool,name='is_trainning')
    with slim.arg_scope(conv_scope):
        train_out=resnet(x_train,10,is_training=is_training,verbose=True)
        valid_out=resnet(x_test,10,is_training=is_training,verbose=True)

    with tf.variable_scope('loss'):
        train_loss=tf.nn.softmax_cross_entropy_with_logits(train_out,y_train)
        # valid_loss=tf.losses.sparse_softmax_cross_entropy(y_test,valid_out,scope='valid')

    with tf.variable_scope('accuracy'):
        with tf.name_scope('train'):
            train_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(train_out, axis=-1, output_type=tf.int32), y_train), tf.float32))

        # with tf.name_scope('valid'):
        #     valid_acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(val_out,axis=-1,output_type=tf.int32),y_test),tf.float32))

with tf.Session() as sess:
    #梯度下降训练
    opt=tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9)
    update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op=opt.minimize(train_out)
        init=tf.initialize_all_variables()
        sess.run(init)
        for i in range(10000):
            operate,loss_value,acc_value=sess.run([update_ops,train_loss,train_acc],feed_dict={is_training:True})
            if i%1000==0:
                print('after {} steps training ,the loss is {},the accuray is {}'.format(i,loss_value,train_acc))


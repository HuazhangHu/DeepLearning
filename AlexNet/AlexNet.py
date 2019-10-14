'''
author ambition
date 
function 
version
'''

'''


'''

import tensorflow as tf
import os
import cv2 as cv
import numpy as np
BATCH_SIZE=64
train_path='./cifar-10/train'
test_path='./cifar-10/test'
MODEL_SAVE_PATH="./model/"#保存训练好的模型
MODEL_NAME="AlexNet_model"

INPUT_SIZE=32
REGULARIZER = 0.0001#正则化系数
LEARNING_RATE_BASE=0.1
NUM_SETS=50000
LEARNING_RATE_DECAY=0.99#学习衰减率
MOVING_AVERAGE_DECAY = 0.99

train_list=os.listdir(train_path)
test_list=os.listdir(test_path)
# img=cv.imread(train_path+'/'+train_list[1])
classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def dataset():
    # writer = tf.python_io.TFRecordWriter("train.tfrecords")
    datasets=[]
    data_labels=[]
    for index, name in enumerate(classes):
        data = []
        label=[]
        class_path = train_path +'/'+ name + '/'
        for i,image_name in enumerate(os.listdir(class_path)):
            image_path=class_path+image_name

            img=cv.imread(image_path)
            # print(img.shape)
            print(type(img))
            data.append(img)

            onehot = []
            hot=classes.index(name)
            for j in range(10):
                if j ==hot:
                    onehot.append(1)
                else:
                    onehot.append(0)

            label.append(onehot)

            if i+1 %BATCH_SIZE==0:

                data=np.reshape(data,[BATCH_SIZE,32,32,3])
                datasets.append(data)
                label=np.reshape(label,[BATCH_SIZE,10])
                print(label)
                data_labels.append(label)
                data=[]
                label=[]

    return datasets,data_labels


def forward(input_tensor, train, regularizer):
    with tf.variable_scope("layer1_conv1"):
        weights_conv1=tf.get_variable("weight",[5,5,3,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biase_conv1=tf.get_variable('bias',[64],dtype=tf.float32,initializer=tf.truncated_normal_initializer(0.0))
        conv1=tf.nn.conv2d(input_tensor,weights_conv1,strides=[1,1,1,1],padding='VALID')
        rule1=tf.nn.relu(tf.nn.bias_add(conv1,biase_conv1))

    with tf.variable_scope('layer2-pool1'):
        pool1=tf.nn.max_pool(rule1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID")

    with tf.variable_scope('layer3-conv2'):
        weights_conv2=tf.get_variable("weight",[5,5,64,64],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        biase_conv2=tf.get_variable('bias',[64],dtype=tf.float32,initializer=tf.truncated_normal_initializer(0.0))
        conv2=tf.nn.conv2d(pool1,weights_conv2,strides=[1,1,1,1],padding='VALID')
        rule2=tf.nn.relu(tf.nn.bias_add(conv2,biase_conv2))

    with tf.variable_scope('layer4-pool2'):
        pool2=tf.nn.max_pool(rule2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID")
        pool_shape=pool2.get_shape().as_list()
        nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
        reshaped=tf.reshape(pool2,[pool2[0],nodes])#pool2[0]代表个数

    with tf.variable_scope('layer5-fc1'):
        weights_fc1=tf.get_variable("weight",[nodes,384],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biase_fc1=tf.get_variable('bias',[384],dtype=tf.float32,initializer=tf.truncated_normal_initializer(0.0))
        if regularizer!=None:
            tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(weights_fc1))
        fc1=tf.nn.relu(tf.matmul(reshaped,weights_fc1)+biase_fc1)
        if train:
            tf.nn.dropout(fc1,0.5)

    with tf.variable_scope('layer6-fc2'):
        weights_fc2=tf.get_variable("weight",[384,192],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biase_fc2=tf.get_variable('bias',[192],dtype=tf.float32,initializer=tf.truncated_normal_initializer(0.0))
        if regularizer!=None:
            tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(weights_fc2))
        fc2=tf.nn.relu(tf.matmul(reshaped,weights_fc2)+biase_fc2)
        if train:
            tf.nn.dropout(fc2,0.5)

    with tf.variable_scope('layer7-fc3'):
        weights_fc3=tf.get_variable("weight",[192,10],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biase_fc3=tf.get_variable('bias',[192],dtype=tf.float32,initializer=tf.truncated_normal_initializer(0.0))
        if regularizer!=None:
            tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(weights_fc3))
        logistic=tf.matmul(fc2,weights_fc3)+biase_fc3

    return logistic

def backward():
    input=tf.placeholder(dtype=tf.float32,shape=[BATCH_SIZE,32,32,3])
    y_=tf.placeholder(dtype=tf.float32,shape=[None,10])
    y=forward(input,train=True,regularizer=REGULARIZER)

    global_step = tf.Variable(0, trainable=False)

    #损失函数定义
    ce=tf.nn.softmax_cross_entropy_with_logits(y,labels=tf.arg_max(y_,-1))
    cem=tf.reduce_mean(ce)
    loss=cem+tf.add_n(tf.add_to_collection("losses"))

    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,NUM_SETS/BATCH_SIZE,LEARNING_RATE_DECAY,staircase=True)
    train_step=tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    #滑动平均
    ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    ema_op=ema.apply(tf.trainable_variables())

    with tf.control_dependencies([train_step,ema_op]):
        train_op=tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)

        ckpt=tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(60000):
            #喂入数据，更新模型，保存模型


            pass

if __name__ == '__main__':
    dataset()

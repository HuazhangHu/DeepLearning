'''
author ambition
date 19.10.14
function 
version

基于keras的AlexNet神经网络
'''

import keras
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from keras.layers import Activation
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras import optimizers
from keras import backend
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) =cifar10.load_data()
x_train = x_train.astype('float')
x_test = x_test.astype('float')
x_train /= 255#归一化、标准化
x_test /= 255

onehot_train = keras.utils.to_categorical(y_train, num_classes=10)
onehot_test = keras.utils.to_categorical(y_test,num_classes=10)
# print(onehot_test.shape,onehot_train.shape)
# print((type(x_train), type(y_train)))
# print((x_train.shape, y_train.shape))

model=Sequential()
#第一层
model.add(Conv2D(64,(5,5),strides=1,padding='valid',input_shape=(32,32,3),activation='relu'))
#第二层
model.add(MaxPool2D(strides=2,pool_size=(2,2)))
#第三层
model.add(Conv2D(64,(5,5),strides=1,padding='valid',activation='relu'))
#第四层
model.add(MaxPool2D([3,3],strides=2))
#进入全连接层
model.add(Flatten())
model.add(Dense(384,activation='relu'))
model.add(Dense(192,activation='relu'))
model.add(Dense(10,activation='softmax'))

# model.summary()
# plot_model(model, to_file='keras_alexnet.png')
# SVG(model_to_dot(model).create(prog='dot', format='svg'))
#配置训练过程
sgd =optimizers.SGD(lr=0.01, momentum=0.9)
#优化方法
model.compile(optimizer=sgd, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])#损失函数,metrics评价函数

sgd = optimizers.SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])
#喂入函数,validation_data(test_x,test_y)验证集
model.fit(x=x_train, y=onehot_train, epochs=25, batch_size=64,validation_data=(x_test,onehot_test))
#评价函数
train_eva=model.evaluate(x_train, onehot_train, batch_size=64)
test_eva=model.evaluate(x_test, onehot_test, batch_size=64)
scroe=model.evaluate(x_test,y_test)
print("test loss:",scroe[0])
print("test accuracy:",scroe[1])




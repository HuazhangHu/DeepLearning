'''
author ambition
date 
function 
version
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(2019)

x_train =np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                   [9.779], [6.182], [7.59], [2.167], [7.042],
                   [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y_train =np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                   [3.366], [2.596], [2.53], [1.221], [2.827],
                   [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


def show(sess):#可视化
    # %matplotlib inline jupyter中使用
    # 输出线性模型
    y_pred_numpy = y_pred.eval(session=sess)

    plt.plot(x_train, y_train, 'bo', label='real')
    plt.plot(x_train, y_pred_numpy, 'ro', label='estimated')
    plt.legend()
    plt.show()

plt.plot(x_train, y_train, 'bo')
x = tf.constant(x_train, name='x')
y = tf.constant(y_train, name='y')

w=tf.Variable(initial_value=tf.random_normal(shape=(),seed=2019),dtype=tf.float32,name="weight")
b=tf.Variable(initial_value=0,dtype=tf.float32,name="biase")

with tf.variable_scope('Linear_Model'):#规定变量域，with中的所有变量都在同一个变量域中，域名为参数
    y_pred=w*x+b
# print(y_pred.name)
# print(w.name)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

# show(sess)

#优化模型
loss=tf.reduce_mean(tf.square(y-y_pred))
print(sess.run(loss))
# print(loss.eval(session=sess))
#两种写法都一样，后者是交互式的session

#求梯度
w_grad,b_grad=tf.gradients(loss,[w,b])#切记用tf返回的都是tensor
# print("w_grad",sess.run(w_grad))
# print("b_grad",sess.run(b_grad))
lr=1e-2#learning rate
w_update=w.assign_sub(lr*w_grad)#w=w-lr*w_grad
b_update=b.assign_sub(lr*b_grad)
sess.run([w_update,b_update])#计算图的运行会更新计算图

# show(sess)

fig=plt.figure()
ax=fig.add_subplot(111)
plt.ion()
fig.show()
fig.canvas.draw()

#训练
sess.run(tf.global_variables_initializer())
for i in range(100):
    sess.run([w_update,b_update])
    y_pred_numpy = y_pred.eval(session=sess)
    loss_numpy=sess.run(loss)
    ax.clear()
    ax.plot(x_train, y_train, 'bo', label='real')
    ax.plot(x_train, y_pred_numpy, 'ro', label='estimated')
    ax.legend()
    fig.canvas.draw()
    plt.pause(0.5)
    print(i,":",loss_numpy)

# show(sess)


#-*- coding:utf-8 -*-
#控制显存使用
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #使用 GPU 0
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session  
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  
set_session(tf.Session(config=config))  

from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout,Activation,Input,Lambda,merge
from keras.layers import Embedding,convolutional
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Reshape
from keras.regularizers import l2
from keras.optimizers import SGD,Adam
from keras.models import load_model

from sklearn import metrics

import math
import numpy as np
import pandas as pd

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_integer('seed', 0, 'Seed for the random generators (>=0)')
#flags.DEFINE_string('name', 'SMS', 'Name for the model.')

# set parameters:=======
#max_features:训练集&测试集中字典中词的总数,max_features的个数=NR_BINS(hash.py中的一个变量)
#maxlen:样本特征的个数
#embedding_dim:embedding的维度
#reg_conf = [0.0001/2,0.00003/2,0.000003/2,0.00003/2,0.0001/2,0.0001/2],分别代码embedding层的正则化系数,3层卷积的正则化系数,全连接层的正则化系数
max_features = 1000000 
maxlen = 22 
batch_size = 1024*10
embedding_dims = 30  
num_epoch = 10
#train_data_num = 40428967
train_data_num = 36000000
test_data_num = 4000000
reg_conf = [0,0,0,0,0,0]
learning_rate = 0.0002

#=======================
#构建迭代器读取数据的辅助函数
#避免数据量太大,导致内存炸掉的情况
def process_line(line):
    tmp = [int(val) for val in line.strip().split(',')]	
    x = np.array(tmp[:-1])
    y = np.array(tmp[-1:])
    return x,y

def load_test_data(path):
    f = open(path)
    X =[]
    Y =[]
    for line in f:
        x, y = process_line(line)
        X.append(x)
        Y.append(y)
    return (np.array(X), np.array(Y))
    f.close()

def generate_arrays_from_file(path,batch_size):
    while 1:
        f = open(path)
        cnt = 0
        X =[]
        Y =[]
        for line in f:
            # create Numpy arrays of input data
            # and labels, from each line in the file
            x, y = process_line(line)
            X.append(x)
            Y.append(y)
            cnt += 1
            if cnt==batch_size:
                cnt = 0
                yield (np.array(X), np.array(Y))
                X = []
                Y = []
    f.close()
###==============

#CCPM的尺寸
m1 = 4 
m2 = 4
w1 = 3
w2 = 3

##====================
#keras构建层的辅助函数
#定义切片操作
def slice(x,index):
        return x[:,:,index]
 
##===========================

#build base
#使用max pooling而不是k-max pooling,实验的结果证明,max pooling的效果略好于k-max pooling
print ('Build model...')

main_input = Input(shape=(maxlen,), dtype='int32')
embedding_map = Embedding(output_dim=embedding_dims, input_dim=max_features,
                          input_length=maxlen,W_regularizer=l2(reg_conf[0]))(main_input)

##
convs = []
for index in range(embedding_dims):
        #print ("i:",index)
        t = Lambda(slice,output_shape=(maxlen,1),arguments={'index':index},
                   name='slice_'+str(index+1))(embedding_map)
        x = Reshape((maxlen,1,1))(t) #(batch, height, width, channels)

        #第一层conv and pooling
        x = Convolution2D(m1,w1,1,border_mode='valid',subsample=(1,1),
                          activation='linear',dim_ordering='tf',W_regularizer=l2(reg_conf[1]),
                          b_regularizer=l2(reg_conf[1]))(x)

        x = MaxPooling2D(pool_size=(2,1),strides=(2,1),
                         border_mode='valid',dim_ordering='tf')(x)
        #x = Activation('tanh')(x)
        x = Activation('sigmoid')(x)

        #第二层conv and pooling
        x = Convolution2D(m2,w2,1,border_mode='valid',subsample=(1,1),
                          activation='linear',dim_ordering='tf',W_regularizer=l2(reg_conf[2]),
                          b_regularizer=l2(reg_conf[2]))(x)
        
        x = MaxPooling2D(pool_size=(2,1),strides=(2,1),
                         border_mode='valid',dim_ordering='tf')(x)
        #x = Activation('tanh')(x)
        x = Activation('sigmoid')(x)        

        flatten = Flatten()(x)
        convs.append(flatten)

merge_map = merge(convs,mode='concat')
x = Dense(embedding_dims,activation='tanh',
          W_regularizer=l2(reg_conf[4]),
          b_regularizer=l2(reg_conf[4]))(merge_map)
# We project onto a single unit output layer, and squash it with a sigmoid:
x = Dense(1,W_regularizer=l2(reg_conf[5]),
          b_regularizer=l2(reg_conf[5]))(x)
main_output = Activation('sigmoid')(x)

model = Model(input=main_input, output=main_output)
adam = Adam(lr =learning_rate)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy','binary_crossentropy'])

#########################

#加载测试集
print ('Loading validation data...')
#test_path = 'Data/val_hash.csv'
test_path = 'Data/data_validation_hash.csv'
(X_test, y_test) = load_test_data(test_path)
print (len(X_test), 'valid sequences')

sample_epochs = math.ceil(train_data_num/batch_size)
print ("samples_per_epoch", sample_epochs)
max_q_size = 10 

#保存测试集预测结果的epoch
save_epoch = [1,3,5,7,9,11,13,15,17,19,20]


def get_AUC(preds,targets):
    test_auc = metrics.roc_auc_score(targets, preds)
    print("AUC:", test_auc)
    with open('Base_auc.txt', 'a+') as f:
        line=str(FLAGS.seed)+":"+str(test_auc)+"\n"
        f.write(line)
        f.close()

#========================
version = 1   
print ("版本号:",version)
print ("batch_size:",batch_size)
print ("embedding_dims:",embedding_dims)
print ("样本特征的域maxlen:", maxlen)
print ("学习率:", learning_rate)
print ("Model_Epochs:", num_epoch)
model_name = 'Base'+'_'+str(version)+'.h5'
#训练的过程,在训练的过程中,生成提交结果
for epoch in range(num_epoch):
    if epoch==0:
        model.fit_generator(generate_arrays_from_file('Data/data_train_hash.csv',batch_size=batch_size),
                            steps_per_epoch=sample_epochs,
                            epochs=2,
                            max_queue_size=max_q_size,
                            verbose=1)
        model.save(model_name)
        print ('finish epoch:',(epoch+1))
    else:
        #加载模型,然后继续训练
        del model
        model = load_model(model_name)
        #这边可以考虑调用函数,每个epoch都将数据随机打乱
        model.fit_generator(generate_arrays_from_file('Data/train_hash.csv',batch_size=batch_size),
                            steps_per_epoch=sample_epochs,
                            epochs=1,
                            max_queue_size=max_q_size,
                            verbose=1)
        model.save(model_name)
        print ('finish epoch:',(epoch+1))
    #在特定的epoch的条件下,生成预测结果
    k = epoch +1
    if k in save_epoch:
        preds = model.predict(X_test)
        get_AUC(preds, y_test)



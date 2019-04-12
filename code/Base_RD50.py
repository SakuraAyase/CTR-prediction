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
num_epoch = 1
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
#定义切片操作cols
def slice(x,index):
        return x[:,:,index,:]

def slice_col(x,index):
        return x[:,:,index]
 
#define slice_row
def slice_row(x,index):
        return x[:,index,:]

##===========================

#build SMS
v1 = [10,21,9,2,4,8,5,13,19,17,22,15,6,7,3,12,16,20,14,18,1,11]
v2 = [13,2,10,3,1,12,8,22,5,16,19,6,15,14,11,17,7,4,21,18,9,20]
v3 = [6,12,21,3,1,19,15,9,2,16,14,11,10,17,8,5,7,22,20,18,4,13]
v4 = [2,7,17,3,9,6,8,15,19,20,16,5,4,18,22,13,11,10,1,12,21,14]
v5 = [17,7,22,18,2,12,5,8,3,1,19,6,14,9,10,4,21,11,16,20,15,13]
v6 = [9,15,12,11,5,19,17,13,7,14,20,18,8,21,3,4,6,10,2,16,1,22]
v7 = [8,20,1,4,17,2,14,19,5,7,16,6,13,10,12,15,11,3,21,22,9,18]
v8 = [8,3,19,4,13,11,22,10,1,18,17,15,20,6,21,14,16,5,2,9,7,12]
v9 = [3,10,20,5,22,19,7,18,17,13,16,21,8,1,4,6,2,14,11,12,9,15]
v10 = [19,22,10,5,14,7,15,3,2,1,12,16,8,20,11,18,21,17,9,6,4,13]
v11 = [4,2,19,3,20,18,13,11,6,14,1,22,21,17,15,10,9,7,8,5,12,16]
v12 = [15,1,17,3,9,7,14,11,22,21,2,16,8,18,20,13,19,6,12,4,5,10]
v13 = [19,10,5,18,6,20,22,3,11,15,12,16,8,13,9,2,4,14,17,7,1,21]
v14 = [4,18,19,10,7,1,17,5,14,11,21,16,9,13,8,22,20,2,3,12,15,6]
v15 = [13,1,12,15,17,14,10,21,18,5,11,16,4,22,8,3,19,6,2,7,20,9]
v16 = [11,14,3,9,21,15,5,1,10,7,22,12,16,2,13,6,18,20,19,4,8,17]
v17 = [11,10,5,7,8,3,18,12,21,9,13,6,16,19,1,2,4,14,15,17,22,20]
v18 = [2,11,13,14,1,21,5,6,9,3,4,20,16,12,18,7,19,22,15,10,8,17]
v19 = [14,4,21,17,18,9,5,13,15,6,7,19,16,1,8,22,3,2,12,11,20,10]
v20 = [2,17,22,11,6,14,8,12,10,15,9,1,20,19,5,13,4,7,3,21,16,18]
v21 = [22,9,15,21,13,4,8,5,17,1,18,2,16,10,14,19,7,6,11,12,20,3]
v22 = [21,7,2,10,15,6,17,9,8,13,4,19,3,16,20,14,5,1,12,22,18,11]
v23 = [3,7,13,6,20,11,1,22,18,8,9,10,17,19,5,4,15,21,16,14,2,12]
v24 = [14,21,9,12,22,11,5,13,19,10,17,3,8,16,7,20,18,1,4,15,2,6]
v25 = [9,1,10,2,3,6,20,15,8,4,19,14,13,16,22,12,17,5,7,21,11,18]
v26 = [8,1,14,6,18,19,9,12,15,2,5,22,10,16,17,21,7,20,3,13,4,11]
v27 = [1,15,4,17,12,2,7,20,19,22,6,9,21,16,8,14,3,5,11,10,18,13]
v28 = [12,17,18,19,6,5,7,1,11,10,14,2,8,4,16,3,13,21,20,22,9,15]
v29 = [9,14,8,10,5,19,13,17,12,20,18,15,4,21,16,2,22,6,7,11,3,1]
v30 = [18,3,8,6,12,17,9,19,4,1,21,13,15,10,22,14,7,5,2,20,11,16]
v31 = [12,22,1,19,9,7,15,4,10,14,20,17,8,6,16,21,18,11,13,5,2,3]
v32 = [1,20,9,8,4,19,18,15,13,10,17,21,12,14,16,7,3,6,5,11,2,22]
v33 = [19,11,1,10,12,4,6,5,21,7,14,8,20,2,18,13,17,16,22,3,9,15]
v34 = [12,1,20,7,18,21,22,14,3,8,2,4,5,6,17,11,16,10,15,19,13,9]
v35 = [4,2,1,22,14,5,3,6,12,21,15,10,18,20,13,16,11,19,9,8,7,17]
v36 = [19,22,13,6,12,8,21,5,1,17,18,15,7,4,11,16,20,2,10,14,3,9]
v37 = [18,1,19,21,3,15,20,17,5,6,22,4,9,2,11,16,13,12,7,14,10,8]
v38 = [4,2,7,14,6,11,15,3,22,12,10,9,13,17,8,16,19,20,5,1,21,18]
v39 = [4,3,7,9,21,22,8,5,20,6,2,17,1,10,15,16,18,11,19,14,13,12]
v40 = [17,21,5,6,14,2,18,8,4,13,1,9,7,3,19,22,10,11,12,20,15,16]
v41 = [16,14,5,3,10,12,1,11,8,20,9,2,22,13,18,17,19,6,21,15,7,4]
v42 = [20,12,21,3,17,10,14,7,11,9,22,19,18,13,5,8,1,2,4,16,6,15]
v43 = [16,5,18,17,15,3,7,1,6,12,10,20,11,19,4,8,14,2,9,22,21,13]
v44 = [16,5,20,13,4,15,7,1,8,19,18,3,6,10,12,14,11,22,17,2,9,21]
v45 = [18,12,6,1,10,14,7,17,5,13,11,19,20,21,22,3,9,16,8,4,15,2]
v46 = [16,13,22,20,9,3,10,2,18,12,1,7,8,19,11,15,5,4,21,6,17,14]
v47 = [5,16,9,6,3,12,18,4,19,10,1,17,2,14,20,21,7,13,15,8,11,22]
v48 = [17,16,12,14,20,7,4,6,9,22,19,21,11,10,18,1,5,3,8,2,13,15]
v49 = [14,16,11,15,18,4,22,3,2,7,12,6,8,1,13,20,21,10,9,17,5,19]
v50 = [2,16,4,11,22,3,20,6,19,7,8,14,18,21,9,5,15,10,13,12,1,17]

v = []
v.append(v1)
v.append(v2)
v.append(v3)
v.append(v4)
v.append(v5)
v.append(v6)
v.append(v7)
v.append(v8)
v.append(v9)
v.append(v10)
v.append(v11)
v.append(v12)
v.append(v13)
v.append(v14)
v.append(v15)
v.append(v16)
v.append(v17)
v.append(v18)
v.append(v19)
v.append(v20)
v.append(v21)
v.append(v22)
v.append(v23)
v.append(v24)
v.append(v25)
v.append(v26)
v.append(v27)
v.append(v28)
v.append(v29)
v.append(v30)
v.append(v31)
v.append(v32)
v.append(v33)
v.append(v34)
v.append(v35)
v.append(v36)
v.append(v37)
v.append(v38)
v.append(v39)
v.append(v40)
v.append(v41)
v.append(v42)
v.append(v43)
v.append(v44)
v.append(v45)
v.append(v46)
v.append(v47)
v.append(v48)
v.append(v49)
v.append(v50)

#使用max pooling而不是k-max pooling,实验的结果证明,max pooling的效果略好于k-max pooling
print ('Build model...')

main_input = Input(shape=(maxlen,), dtype='int32')
embedding_map = Embedding(output_dim=embedding_dims, input_dim=max_features,
                          input_length=maxlen,W_regularizer=l2(reg_conf[0]))(main_input)
em_map = Reshape((maxlen, embedding_dims, 1))(embedding_map)

SliceM = []
for ind in range(maxlen):
        tmp = Lambda(slice_row,output_shape=(1,embedding_dims),arguments={'index':ind},
                     name='slice_row_'+str(ind+1))(embedding_map)
        SliceM.append(tmp)

Base_RD = []
vt = v[FLAGS.seed]
for ind in range(maxlen):
        t = vt[ind]
        Base_RD.append(SliceM[t-1])
Base_RD_mg = merge(Base_RD,mode='concat',concat_axis=1)
Base_RD_map = Reshape((maxlen, embedding_dims, 1))(Base_RD_mg)

##
convs = []
for index in range(embedding_dims):
        #print ("i:",index)
        t = Lambda(slice,output_shape=(maxlen,1,1),arguments={'index':index},
                   name='slice_'+str(index+1))(Base_RD_map)
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
    with open('BaseRD_auc.txt', 'a+') as f:
        line=str(FLAGS.seed)+":"+str(test_auc)+"\n"
        f.write(line)
        f.close()


#============================
version = 1   
print ("版本号:",version)
print ("batch_size:",batch_size)
print ("embedding_dims:",embedding_dims)
print ("样本特征的域maxlen:", maxlen)
print ("学习率:", learning_rate)
model_name = 'Base_RD'+'_'+str(version)+'.h5'
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
        model.fit_generator(generate_arrays_from_file('Data/data_train_hash.csv',batch_size=batch_size),
                            steps_per_epoch=sample_epochs,
                            epochs=2,
                            max_queue_size=max_q_size,
                            verbose=1)
        model.save(model_name)
        print ('finish epoch:',(epoch+1))
    #在特定的epoch的条件下,生成预测结果
    k = epoch +1
    if k in save_epoch:
        preds = model.predict(X_test)
        get_AUC(preds, y_test)

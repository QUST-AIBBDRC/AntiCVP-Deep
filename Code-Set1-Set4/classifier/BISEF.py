
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import re
from scipy import interp
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, merge, Reshape, Lambda
from keras.layers import Flatten
from keras.layers import LSTM, GRU, Bidirectional
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale, StandardScaler
from keras.layers import Dense, merge, Input, Dropout
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D
from tensorflow.python.keras.layers import Layer, InputSpec
from keras import backend as K
import utils.tools as utils





    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        print("WQ.shape", WQ.shape)

        print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(
            WK, [0, 2, 1]).shape)

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (self.output_dim**0.5)

        QK = K.softmax(QK)

        print("QK.shape", QK.shape)

        V = K.batch_dot(QK, WV)

        return V

    def compute_output_shape(self, input_shape):

        return (input_shape[0], input_shape[1], self.output_dim)


# time_step = 10
# features_num = 1100
# seg_len = int(features_num / time_step)

def to_class(p):
    return np.argmax(p, axis=1)

def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y
# Origanize data
def get_shuffle(dataset,label):    
    #shuffle data
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset,label 
# Origanize data

data_=pd.read_csv(r'KMeansSMOTE_Training_Set1_fusion.csv')
data=np.array(data_)
data=data[:,1:]
[m1,n1]=np.shape(data)
#label1=np.ones((int(m1/2),1))#Value can be changed
#label2=np.zeros((int(m1/2),1))
label1=np.ones((1399,1))#Value can be changed
label2=np.zeros((1399,1))
label=np.append(label1,label2)
X_=scale(data)
y_= label
X,y=get_shuffle(X_,y_)
sepscores = []
sepscores_ = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5



def get_RNN_model(input_dim,out_dim):
    model = Sequential()
    model.add(Bidirectional(LSTM(int(input_dim/2), return_sequences=True),input_shape=(1,input_dim)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(int(input_dim/4), return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Self_Attention(64))
    model.add(Flatten())
    model.add(Dense(int(input_dim/4), activation = 'sigmoid'))
    model.add(Dense(int(input_dim/8), activation = 'sigmoid'))
    model.add(Dense(out_dim, activation = 'softmax',name="Dense_2"))
    model.compile(loss = 'binary_crossentropy', optimizer = 'SGD', metrics =['accuracy'])
    return model

[sample_num,input_dim]=np.shape(X)
out_dim=2
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5
probas_rnn=[]
tprs_rnn = []
sepscore_rnn = []
skf= StratifiedKFold(n_splits=5)
for train, test in skf.split(X,y):
    time_step = 10
    features_num = input_dim
    seg_len = int(features_num / time_step)
    clf_rnn = get_RNN_model(input_dim,out_dim)
    X_train_rnn=np.reshape(X[train],(-1,1,input_dim))
    X_test_rnn=np.reshape(X[test],(-1,1,input_dim))
    clf_list = clf_rnn.fit(X_train_rnn, to_categorical(y[train]),batch_size=16, epochs=80)
    y_rnn_probas=clf_rnn.predict(X_test_rnn)
    probas_rnn.append(y_rnn_probas)
    y_class= utils.categorical_probas_to_classes(y_rnn_probas)
    y_test=utils.to_categorical(y[test])#generate the test 
    ytest=np.vstack((ytest,y_test))
    y_test_tmp=y[test]  
    yscore=np.vstack((yscore,y_rnn_probas))
    
    acc, precision,npv, sensitivity, specificity, mcc,f1, f2, Gmean = utils.calculate_performace(len(y_class), y_class,y[test])
    mean_fpr = np.linspace(0, 1, 150)
    fpr, tpr, thresholds = roc_curve(y[test], y_rnn_probas[:, 1])
    tprs_rnn.append(interp(mean_fpr, fpr, tpr))
    tprs_rnn[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    sepscore_rnn.append([acc, precision,npv, sensitivity, specificity, mcc,f1, f2, Gmean, roc_auc])
                            

row=ytest.shape[0]
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('ytest_sum_self_KMeans_Train_Set1_80.csv')

yscore_=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore_)
yscore_sum.to_csv('yscore_sum_self_KMeans_Train_Set1_80.csv')

scores=np.array(sepscore_rnn)
result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscore_rnn.append(H1)
result=sepscore_rnn
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('self_KMeans_Train_Set1_80.csv')


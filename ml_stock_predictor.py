import pandas as pd
import quandl
import math, datetime
import numpy as np
import time
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
#import os
#import io
#import tempfile

#linearregression = tempfile.mktemp(dir=linearregression)
#os.makedirs(linearregression)
#os.rmdir(linearregression)

#def mkdirp(linearregression):
    #if not os.path.isdir(linearregression):
        #os.makedirs(linearregression) 
#from app.stamp import Timestamp

style.use('ggplot')

df = quandl.get('WIKI/GOOGl')
df= df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['hl_pct'] = ((df['Adj. High'] - df['Adj. Low'])/ df['Adj. Low'])*100
df['co_pct'] = ((df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'])*100
df= df[['Adj. Close','hl_pct','co_pct','Adj. Volume']]
predict_col= 'Adj. Close'
df.fillna(-99999, inplace=True)  #try experimenting with different values
predict_num= int(math.ceil(0.1*len(df)))    #try experimenting with different values
df['label']= df[predict_col].shift(-predict_num)

X= np.array(df.drop(['label'],1))#this is gonna be our features
X= preprocessing.scale(X)
X_latest= X[-predict_num:]
X= X[:-predict_num]

df.dropna(inplace=True)
y= np.array(df['label'])#this is gonnna be our prediction label
          

#X= X[:-predict_num+1] #array index starts from 0. X[4]= C[5] where predict_num = 1
#df.dropna(inplace= True)
#print(len(X),len(y))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#clf= svm.SVR(kernel='poly')
clf= LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

#filename= LinearRegression
#dirname = os.path.dirname(filename)
#if not os.path.exists(dirname):
    #os.makedirs(dirname)
#filename = linearregression.pickle
#dirname = os.path.dirname(filename)
#if not os.path.exists(dirname):
    #os.makedirs(dirname)


with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)                            #serializing clf
    
pickle_open= open('linearregression.pickle','rb')
clf= pickle.load(pickle_open)                                   #deserializing clf

    

accuracy = clf.score(X_test, y_test)
predict_set= clf.predict(X_latest)
df['Forecast']= np.nan

#print(predict_set,accuracy, predict_num)

last_date= df.iloc[-1].name
last_unix= last_date.timestamp() 


next_unix= last_unix + 86400



for i in predict_set:                                                 #google it
    next_date= datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date]= [np.nan for _ in range(len(df.columns)-1)] + [i]
    
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('date')
plt.ylabel('price')
plt.show()

















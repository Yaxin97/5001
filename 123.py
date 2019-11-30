import os
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR,SVC
import warnings
warnings.filterwarnings('ignore')

os.chdir('/Users/yaxin/Google Drive/BDT/5001/Ind-project')
train = pd.read_csv("./data/train.csv", parse_dates=["purchase_date","release_date"])
test = pd.read_csv("./data/test.csv", parse_dates=["purchase_date","release_date"])

def mysplit(data):
    x = []
    for i in range(len(data)):
        thisone = data.iloc[i].split(",")
        x.append(thisone)
    return x
def myremove(text):
    remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    return re.sub(remove_chars, '', text)
def tomystring(x):
    for n in range(len(x)):
        line = x[n]
        for i in range(len(line)):
            line[i] = myremove(line[i])
        catstr = ' ' # 改为用 分割的长字符串
        x[n] = catstr.join(line)
    return x
def tomatrix(corpus):
    vector = TfidfVectorizer()
    tf_data = vector.fit_transform(corpus)
    #print(tf_data)    #(句子下标, 单词特征下标)   权重
    #print(vector.vocabulary_)    #单词特征
    df1 = pd.DataFrame(tf_data.toarray(), columns=vector.get_feature_names()) # to DataFrame
    return df1
def preprocessing(Features,splitloc):
    #nan
    data = Features.copy()
    data.loc[(data.purchase_date.isnull()),'purchase_date'] = data.loc[(data.purchase_date.isnull()),'release_date']
    data["total_positive_reviews"].fillna(data.total_positive_reviews.mean(),inplace=True)
    data["total_negative_reviews"].fillna(data.total_negative_reviews.mean(),inplace=True)
    #date
    data['purchase_date'] = data['purchase_date'].apply(lambda x:time.mktime(x.timetuple()))
    data['release_date'] = data['release_date'].apply(lambda x:time.mktime(x.timetuple()))
    #is_free
    data["is_free"].replace(False,0,inplace=True)
    # genres,categories,tags
    genres = mysplit(data["genres"])
    categories = mysplit(data["categories"])
    tags = mysplit(data["tags"])
    genresstr = tomystring(genres)
    categoriesstr = tomystring(categories)
    tags = tomystring(tags)
    gen = tomatrix(genres)
    cat = tomatrix(categories)
    tag = tomatrix(tags)
    data.drop(["genres","categories","tags"],axis=1,inplace=True)
    data = pd.concat([data, gen,cat,tag], axis=1,join_axes=[data.index])
    #shuffle
    data = shuffle(data)
    data = shuffle(data)
    #scale
    data = scale(data)
    #pca
    pca=PCA(n_components=26)
    data=pca.fit_transform(data)
    #split
    traindata = data[:splitloc,:]
    testdata = data[splitloc:,:]
    return(traindata,testdata)
def ycls(yc):
    yc1 = yc[:]
    for i in range(len(yc)):
        if yc1[i] == 0:
            yc1[i] = 0
        elif yc1[i] < 2:
            yc1[i] = 1
        elif yc1[i] < 40:
            yc1[i] = 2
        else:
            yc1[i] = 3
    return yc1
def clsreg(y_trc,X_train,y_train):
    y0 = []
    y1 = []
    y2 = []
    y3 = []
    x0 = []
    x1 = []
    x2 = []
    x3 = []
    for i in range(len(y_trc)):
        if y_trc[i] == 0:
            x0.append(X_train[i])
            y0.append(y_train[i])
        elif y_trc[i] == 1:
            x1.append(X_train[i])
            y1.append(y_train[i])
        elif y_trc[i] == 2:
            x2.append(X_train[i])
            y2.append(y_train[i])
        else:
            x3.append(X_train[i])
            y3.append(y_train[i])
    x0 = np.array(x0)
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    reg0 = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), {"alpha": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)},cv=5)
    reg1 = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), {"alpha": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)},cv=5)
    reg2 = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), {"alpha": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)},cv=5)
    reg3 = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), {"alpha": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)},cv=5)
    reg0.fit(x0,y0)
    reg1.fit(x1,y1)
    reg2.fit(x2,y2)
    reg3.fit(x3,y3)
    return reg0,reg1,reg2,reg3
def myprey(X_test,pre_cls,reg0,reg1,reg2,reg3):
    yp = []
    for i in range(len( X_test)):
        if pre_cls[i] == 0:
            p = reg0.predict(np.array([X_test[i]]))
            yp.append(list(p)[0])
        elif pre_cls[i] == 1:
            p = reg1.predict(np.array([X_test[i]]))
            yp.append(list(p)[0])
        elif pre_cls[i] == 2:
            p = reg2.predict(np.array([X_test[i]]))
            yp.append(list(p)[0])
        else:
            p = reg3.predict(np.array([X_test[i]]))
            yp.append(list(p)[0])
    return yp
def myoutput(train,test):
    X_train,X_test = preprocessing(pd.concat([train.drop(["playtime_forever"],axis=1),test]),len(train))
    y_train = list(train["playtime_forever"])
    y_trc = ycls(y_train)
    rfc=RandomForestClassifier(max_features='auto', n_estimators= 200, max_depth=4, criterion='gini')
    rfc.fit(X_train,y_trc)
    pre_cls = rfc.predict(X_test)
    reg0,reg1,reg2,reg3 = clsreg(y_trc,X_train,y_train)
    output = myprey(X_test,pre_cls,reg0,reg1,reg2,reg3)
    return output

playtime = pd.DataFrame({"playtime_forever":myoutput(train,test)})
playtime = pd.concat([test.id,playtime],axis=1)
playtime.set_index(["id"], inplace=True)
playtime.to_csv('playtime.csv')
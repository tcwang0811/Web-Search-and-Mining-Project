import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from scipy.sparse import csr_matrix, hstack
import xgboost as xgb
import gc
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import os

def bm25(corpus,b,k1, stopword):
    CV = CountVectorizer(ngram_range=(1,1), stop_words = stopword, min_df=5,max_df=0.3)
    IDFTrans = TfidfTransformer(norm='l2')
    
    output = CV.fit_transform(corpus)
    IDFTrans.fit(output)
    temp = output.copy()
    
    aveL = output.sum()/output.shape[0]
    denominator = k1 * ((1-b)+b*(output.sum(1)/aveL))
    
    temp.data = temp.data/temp.data
    temp = csr_matrix.multiply(temp,denominator)
    
    temp += output
    output *= (k1+1)

    temp.data = 1/temp.data
    output = csr_matrix.multiply(output,temp)
    
    output = IDFTrans.transform(output)
    
    return output
	
sw = stopwords.words('russian')

train =pd.read_csv("./data/avito-demand-prediction/train.csv") 
test =pd.read_csv("./data/avito-demand-prediction/test.csv")

dummy = ["parent_category_name","user_type","region","category_name"] # dummy variable directly
categorical = ["user_id","city"] # labelencoding => one-hot
nullP = ["image_top_1","param_1","param_2","param_3"] # labelencoding with NA (add an indicator to identify whether it is NA)  => one-hot
dropOr = ["item_id","title","description"] # to drop

trainIndex=train.shape[0]
train_y = train.deal_probability
train_x = train.drop(columns="deal_probability")

tr_te = pd.concat([train_x,test],axis=0)

tr_te = tr_te.assign(mon=lambda x: pd.to_datetime(x['activation_date']).dt.month,
                     mday=lambda x: pd.to_datetime(x['activation_date']).dt.day,
                     week=lambda x: pd.to_datetime(x['activation_date']).dt.week,
                     wday=lambda x:pd.to_datetime(x['activation_date']).dt.dayofweek,
                     txt=lambda x:(x['title'].astype(str)+' '+x['description'].astype(str)))

del train, test, train_x
gc.collect()

tr_te["price"] = np.log(tr_te["price"]+0.001)
tr_te["price"].fillna(tr_te.price.mean(),inplace=True)

tr_te.drop(["activation_date","image"],axis=1,inplace=True)

# labelencoding with NA
lbl = preprocessing.LabelEncoder()
for col in nullP:
    toApp = tr_te[col].isnull()
    tr_te[col].fillna('Unknown')
    tr_te[col] = lbl.fit_transform(tr_te[col].astype(str))
    toApp *= 1
    theName = "isNA_" + col
    tr_te = pd.concat([tr_te,toApp.rename(theName)],axis=1)
	
# labelencoding
for col in categorical:
    tr_te[col].fillna('Unknown')
    tr_te[col] = lbl.fit_transform(tr_te[col].astype(str))
	
# dummy
for col in dummy:
    temp = pd.get_dummies(tr_te[col],prefix = col)
    tr_te.drop(columns=col,inplace=True)
    tr_te = pd.concat([tr_te,temp],axis=1)
	
oneHot = categorical+nullP
oneHot.remove("user_id")

ohe = preprocessing.OneHotEncoder()
sparseOneHot = ohe.fit_transform(tr_te[oneHot])

del ohe, lbl
gc.collect()

tr_te.drop(labels=dropOr,axis=1,inplace=True)
tr_te.drop(labels=oneHot,axis=1,inplace=True)

tr_te.loc[:,'txt']=tr_te.txt.apply(lambda x:x.lower().replace("[^[:alpha:]]"," ").replace("\\s+", " "))

m_tfidf = bm25(tr_te.txt,0.75,2,stopword=sw)

tr_te.drop(labels=['txt'],inplace=True,axis=1)

data  = hstack((tr_te.values, sparseOneHot, m_tfidf)).tocsr()

del tr_te, m_tfidf, sparseOneHot
gc.collect()

dtest = xgb.DMatrix(data=data[trainIndex:])
train = data[:trainIndex]

del data
gc.collect()

X_train, X_valid, y_train, y_valid = train_test_split(train, train_y,test_size = 0.1, random_state=5566)

del train, train_y
gc.collect()

dtrain = xgb.DMatrix(data = X_train, label=y_train)
deval = xgb.DMatrix(data = X_valid, label=y_valid)
watchlist = [(deval, 'eval')]

del X_train, X_valid, y_train, y_valid
gc.collect()

Dparam = {'objective' : "reg:logistic",
          'booster' : "gbtree",
          'eval_metric' : "rmse",
          'nthread' : 4,
          'eta':0.05,
          'max_depth':18,
          'min_child_weight': 11,
          'gamma' :0,
          'subsample':0.8,
          'colsample_bytree':0.7,
          'aplha':2.25,
          'lambda':0,
          'nrounds' : 5000}
		  
xgb_clf = xgb.train(params=Dparam,dtrain=dtrain,num_boost_round=Dparam['nrounds'],early_stopping_rounds=50,evals=watchlist,verbose_eval=10)

pd.read_csv("./data/avito-demand-prediction/sample_submission.csv").assign(deal_probability = xgb_clf.predict(dtest)).to_csv("XGB_BM25.csv", index=False)
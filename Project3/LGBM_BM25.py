import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from scipy.sparse import csr_matrix, hstack
import lightgbm as lgb
import gc
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import re

import os
# print(os.listdir("./data/imagefeature/"))

def bm25(corpus,b,k1, stopword):
    CV = CountVectorizer(ngram_range=(1,1), stop_words = stopword, min_df=5,max_df=0.3)
    IDFTrans = TfidfTransformer(norm='l2')
    
    output = CV.fit_transform(corpus)
    IDFTrans.fit(output)
    feature_names = CV.get_feature_names()
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
    
    return output, feature_names
	
def cleanName(text):
    try:
        textProc = text.lower()
        textProc = re.sub('[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]', '', textProc)
        textProc = " ".join(textProc.split())
        return textProc
    except: 
        return "name error"
		
sw = stopwords.words('russian')

train =pd.read_csv("./data/avito-demand-prediction/train.csv") 
test =pd.read_csv("./data/avito-demand-prediction/test.csv")
image_train = pd.read_csv("./data/imagefeature/train_quality.csv")
image_test = pd.read_csv("./data/imagefeature/test_quality.csv")

image_quality = pd.concat([image_train,image_test],axis=0)

del image_train, image_test
gc.collect()

# check NA before adding
categorical = ["user_id","city","parent_category_name","user_type","region","category_name"] # labelencoding
nullP = ["image_top_1","param_1","param_2","param_3"] # labelencoding with NA (add an indicator to identify whether it is NA)
isNA = [] # indicator of NA
dropOr = ["item_id"] # to drop
textFeature = ["title","description"]

trainIndex=train.shape[0]
train_y = train.deal_probability
train_x = train.drop(columns="deal_probability")

tr_te = pd.concat([train_x,test],axis=0)
tr_te = pd.merge(tr_te,image_quality,how="left")

tr_te = tr_te.assign(mon=lambda x: pd.to_datetime(x['activation_date']).dt.month,
                     mday=lambda x: pd.to_datetime(x['activation_date']).dt.day,
                     week=lambda x: pd.to_datetime(x['activation_date']).dt.week,
                     wday=lambda x:pd.to_datetime(x['activation_date']).dt.dayofweek,
                     isNA_description = lambda x: x['description'].isnull()*1,
                     title=lambda x:(x['title'].astype(str)),
                     description=lambda x:(x['description'].astype(str)))

isNA.append("isNA_description")

del train, test, train_x, image_quality
gc.collect()

tr_te["price"] = np.log(tr_te["price"]+0.001)
tr_te["price"].fillna(tr_te.price.mean(),inplace=True)

tr_te.drop(["activation_date","image"],axis=1,inplace=True)

imageNA = ["dullness","whiteness","average_pixel_width","dominant_red","dominant_green","dominant_blue","average_red","average_green","average_blue","image_size","width","height","blurrness"]
for col in imageNA:
    tr_te[col].fillna(tr_te[col].mean(),inplace=True)
	
# labelencoding with NA
lbl = preprocessing.LabelEncoder()
for col in nullP:
    toApp = tr_te[col].isnull()
    tr_te[col].fillna("Unknown",inplace = True)
    tr_te[col] = lbl.fit_transform(tr_te[col].astype(str))
    toApp *= 1
    theName = "isNA_" + col
    isNA.append(theName)
    tr_te = pd.concat([tr_te,toApp.rename(theName)],axis=1)
	
# labelencoding
for col in categorical:
    tr_te[col].fillna('Unknown')
    tr_te[col] = lbl.fit_transform(tr_te[col].astype(str))
	
tr_te.drop(labels=dropOr,axis=1,inplace=True)

tr_te.loc[:,'title']=tr_te.title.apply(lambda x:x.lower().replace("[^[:alpha:]]"," ").replace("\\s+", " "))
tr_te.loc[:,'description']=tr_te.description.apply(lambda x:x.lower().replace("[^[:alpha:]]"," ").replace("\\s+", " "))

for text in textFeature:
    tr_te[text] = tr_te[text].apply(lambda x: cleanName(x))
    tr_te[text] = tr_te[text].astype(str).fillna('missing')
    tr_te[text + '_num_words'] = tr_te[text].apply(lambda comment: len(comment.split())) # Count number of Words
    tr_te[text + '_num_unique_words'] = tr_te[text].apply(lambda comment: len(set(w for w in comment.split())))
    tr_te[text + '_words_vs_unique'] = tr_te[text+'_num_unique_words'] / tr_te[text+'_num_words'] * 100 # Count Unique Words
    tr_te[text + '_num_letters'] = tr_te[text].apply(lambda comment: len(comment)) # Count number of Letters
    
tr_te['title_desc_len_ratio'] = tr_te['title_num_letters']/tr_te['description_num_letters']

tr_te["txt"] = tr_te['title'] +" "+ tr_te['description']
tr_te.drop(labels=textFeature,axis=1,inplace=True)

m_tfidf, tfidf_feature = bm25(tr_te.txt,0.75,2,stopword=sw)

tr_te.drop(labels=['txt'],inplace=True,axis=1)

feature_list = tr_te.columns.values.tolist()
feature_list.extend(tfidf_feature)

categorical.extend(nullP)
categorical.extend(isNA)

data  = hstack((tr_te.values,m_tfidf)).tocsr()

del tr_te,m_tfidf
gc.collect()

dtest = data[trainIndex:]
train = data[:trainIndex]

del data
gc.collect()

X_train, X_valid, y_train, y_valid = train_test_split(train, train_y,test_size = 0.1, random_state=5566)

del train, train_y
gc.collect()

dtrain = lgb.Dataset(X_train,y_train,
                     feature_name=feature_list,
                     categorical_feature = categorical)
deval = lgb.Dataset(X_valid,y_valid,
                     feature_name=feature_list,
                     categorical_feature = categorical)

del X_train, X_valid, y_train, y_valid
gc.collect()

lgbm_params =  {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 270,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.75,
    'bagging_freq': 2,
    'learning_rate': 0.016,
    'verbose': 0
}  

lgb_clf = lgb.train(
        lgbm_params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[deval],
        valid_names=['valid'],
        early_stopping_rounds=50,
        verbose_eval=10
    )
	
print("Predicting")
dpred = lgb_clf.predict(dtest)

output = pd.read_csv("./data/avito-demand-prediction/sample_submission.csv")
output['deal_probability'] = dpred
output['deal_probability'].clip(0.0, 1.0, inplace=True)
output.to_csv("lgbm_7.csv", index=False)
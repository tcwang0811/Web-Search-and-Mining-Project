from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import csr_matrix

def bm25(corpus, b, k1, stopword):
    CV = CountVectorizer(ngram_range=(1,1), stop_words = stopword, min_df=5,max_df=0.3,max_features=5000)
    IDFTrans = TfidfTransformer(norm='l2')
    
    output = CV.fit_transform(corpus)
    IDFTrans.fit(output)
    temp = output.copy()
    
    aveL = output.sum()/output.shape[0]
    denominator = k1 * ((1-b)+b*(output.sum(1)/aveL))
    
	# set elements of every row to k1*((1-b)+b*(docL/aveL))
    temp.data = temp.data/temp.data
    temp = csr_matrix.multiply(temp,denominator)
    
	# to tf + k1*((1-b)+b*(docL/aveL))
    temp += output
    output *= (k1+1)
	
	# reciprocal and then multiply
    temp.data = 1/temp.data
    output = csr_matrix.multiply(output,temp)
    
    output = IDFTrans.transform(output)
    
    return output

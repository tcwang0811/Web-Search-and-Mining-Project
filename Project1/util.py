import sys

# http://www.scipy.org/
try:
    import numpy as np
    from numpy.linalg import norm
except:
    print("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
    sys.exit()


def removeDuplicates(list):
    """ remove duplicates from a list """
    return set((item for item in list))

def normalization(list):
    normList = norm(list)

    if normList == 0:
        return list
    else:
        return list/normList

def cosine(vector1, vector2):
    """ related documents j and q are in the concept space by comparing the vectors :
		cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
    return float(np.dot(vector1, vector2) / (norm(vector1) * norm(vector2)))

def EuclideanDist(document, query):
    Doc = np.array(document)
    Que = np.array(query)

    return float(norm(Doc-Que))

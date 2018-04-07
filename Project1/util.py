import sys

# http://www.scipy.org/
try:
    from numpy import dot
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
    return float(dot(vector1, vector2) / (norm(vector1) * norm(vector2)))

def EuclideanDist(document, query):
    output = [0]*len(document)

    for i in range(len(document)):
        output[i] = document[i] - query[i]

    return float(norm(output))
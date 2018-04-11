import os
import nltk
from VectorSpaceModel import VectorSpaceModel

def SandP(VSList, NoList, n, isReverse):
    print("DocID  ", "Score")

    output = []

    for i in range(len(NoList)):
        output.append([NoList[i], round(VSList[i], 6)])

    output.sort(key = lambda x: x[1],reverse = isReverse)

    for i in range(n):
        print(output[i][0][:6],output[i][1])

    return output

def NandV(documentString):
    text = nltk.word_tokenize(documentString)
    temp = nltk.pos_tag(text)

    output = ""

    for i in range(len(temp)):
        if temp[i][1][0] == "N" or temp[i][1][0] == "V":
            if output != "":
                output += " "
            output += temp[i][0]

    return output


def main():

    # open and store document
    path = "documents/"

    documents = []
    documentsNo = []

    for file in os.listdir(path):
        if file.endswith(".product"):
            thisFile = open(path + file, 'r')
            documents.append(thisFile.read())
            documentsNo.append(file)
            thisFile.close()

    query = ["drill wood sharp"]

    VS = VectorSpaceModel(documents)

    n_Print = 5

    #######################################
    ## TF + Cosine ##

    print("TF + Cosine:")

    SandP(VS.searchTFWithCosine(query), documentsNo, n_Print,True)

    print("\n")
    ## TF + Euclidean Distance ##
    print("TF + Euclidean Distance:")

    SandP(VS.searchTFWithEuclideanDist(query), documentsNo, n_Print, False)

    print("\n")
    ## TF-IDF + Cosine ##
    print("TF-IDF + Cosine:")

    FBVector = SandP(VS.searchTFIDFWithCosine(query), documentsNo, n_Print, True)

    print("\n")
    ## TF-IDF + Euclidean Distance ##
    print("TF-IDF + Euclidean Distance:")

    SandP(VS.searchTFIDFWithEuclideanDist(query), documentsNo, n_Print, False)

    print("\n")

    ## Relevant Feedback ##
    print("TF-IDF + Cosine + Relevant Feedback:")
    FBNo = FBVector[0][0]
    FB = documents[documentsNo.index(FBNo)]

    FB = NandV(FB)

    newQuery = [query,[FB]]

    SandP(VS.searchRelevantFeedback(newQuery), documentsNo, n_Print, True)


    #######################################

if __name__ == "__main__":
    main()
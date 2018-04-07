import os
from VectorSpaceModel import VectorSpaceModel

def SandP(VSList, NoList, n, isReverse):
    print("DocID  ", "Score")

    output = []

    for i in range(len(NoList)):
        output.append([NoList[i][:6], round(VSList[i], 6)])

    output.sort(key = lambda x: x[1],reverse = isReverse)

    for i in range(n):
        print(output[i][0],output[i][1])


def main():

    # open and store document
    path = "D:/PycharmProjects/WSMProject/documents/"

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

    ## TF + Cosine
    print("TF + Cosine:")

    SandP(VS.searchTFWithCosine(query), documentsNo, n_Print,True)

    print("\n")
    ## TF + Euclidean Distance
    print("TF + Euclidean Distance:")

    SandP(VS.searchTFWithEuclideanDist(query), documentsNo, n_Print, False)

    print("\n")
    ## TF-IDF + Cosine
    print("TF-IDF + Cosine:")

    SandP(VS.searchTFIDFWithCosine(query), documentsNo, n_Print, True)

    print("\n")
    ## TF-IDF + Euclidean Distance
    print("TF-IDF + Euclidean Distance:")

    SandP(VS.searchTFIDFWithEuclideanDist(query), documentsNo, n_Print, False)

    print("\n")

    #######################################

if __name__ == "__main__":
    main()
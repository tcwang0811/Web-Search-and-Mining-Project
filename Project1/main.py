import os
from VectorSpaceModel import VectorSpaceModel

def PrintFormat(SortedList, n):
    for i in range(n):
        print(SortedList[i][0],SortedList[i][1])


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

    query = "drill wood sharp"

    VS = VectorSpaceModel(documents)


    n_Print = 5

    ## TF + Cosine
    # print("TF + Cosine:")
    # print("DocID  ", "Score")
    #
    # TFWithCosine = VS.searchTFWithCosine(query)
    #
    # output = []
    #
    # for i in range(len(documentsNo)):
    #     output.append([documentsNo[i][:6],round(TFWithCosine[i],6)])
    #
    # output.sort(key=lambda x: x[1],reverse=True)
    #
    # PrintFormat(output, n_Print)

    # TFWithEuclideanDist = VS.searchTFWithEuclideanDist(query)
    # TFIDFWithCosine = VS.searchTFIDFWithCosine(query)
    # TFIDFWithEuclideanDist = VS.searchTFIDFWithEuclideanDist(query)


if __name__ == "__main__":
    main()
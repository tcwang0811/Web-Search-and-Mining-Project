import os
from VectorSpaceModel import VectorSpaceModel



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





if __name__ == "__main__":
    main()
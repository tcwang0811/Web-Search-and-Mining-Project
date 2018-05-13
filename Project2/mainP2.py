from Models import Model


def main():
    # changeable
    pathData = "data/toTest.txt"
    pathID = "data/ID/ID_toTest.txt"
    pathQuery = "data/Query/toTest_query.txt"
    uniqueTerm = 27439  # 不在檔案中
    isStemming = False

    # declare model
    MD = Model(pathData=pathData, pathID=pathID, pathQuery=pathQuery, uniqueT=uniqueTerm, isStemming=isStemming)


    print("VectorSpace")
    MD.printVectorSpace()

    print("")
    print("Laplace")
    MD.printLanguageModelLaplace()

    print("")
    print("JM")
    MD.printLanguageModelJM()


if __name__ == '__main__':
    main()

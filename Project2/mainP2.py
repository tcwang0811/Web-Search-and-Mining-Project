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

    print("VectorSpace\n")
    MD.printVectorSpace()
    print("Laplace\n")
    MD.printLanguageModelLaplace()
    print("JM\n")
    MD.printLanguageModelJM()


if __name__ == '__main__':
    main()

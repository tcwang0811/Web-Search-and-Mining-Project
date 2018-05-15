from Models import Model


def main():
    # changeable

    print("Initialize")

    pathData = "data/toTest.txt"
    pathID = "data/ID/ID_toTest.txt"
    pathQuery = "data/Query/toTest_query.txt"
    uniqueTerm = 1525012  # 不在檔案中
    isStemming = False

    # declare model
    MD = Model(pathData=pathData, pathID=pathID, pathQuery=pathQuery, uniqueT=uniqueTerm, isStemming=isStemming)

    # print(MD._Model__index)
    # print(MD._Model__query)
    # print(MD._Model__querySet)

    print("VectorSpace Start")

    VSFile = open("output/wo_stemming/wo_stemming_VSFile.txt","wt")
    MD.printVectorSpace(VSFile)
    VSFile.close()

    print("VectorSpace End")

    print("")
    print("Laplace Start")

    LaplaceFile = open("output/wo_stemming/wo_stemming_LaplaceFile.txt", "wt")
    MD.printLanguageModelLaplace(LaplaceFile)
    LaplaceFile.close()

    print("Laplace End")

    print("")
    print("JM Start")

    JMFile = open("output/wo_stemming/wo_stemming_JMFile.txt", "wt")
    MD.printLanguageModelJM(JMFile)
    JMFile.close()

    print("JM End")


if __name__ == '__main__':
    main()

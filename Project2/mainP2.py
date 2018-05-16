from Models import Model


def main():
    # changeable

    print("Initialize")

    pathData = "data/Porter.txt"
    pathID = "data/ID/ID_Porter.txt"
    pathQuery = "data/Query/query.txt"
    uniqueTerm = 1341890  # 不在檔案中
    isStemming = True

    # declare model
    MD = Model(pathData=pathData, pathID=pathID, pathQuery=pathQuery, uniqueT=uniqueTerm, isStemming=isStemming)

    print("VectorSpace Start")

    VSFile = open("output/Porter/Porter_VSFile.txt","wt")
    MD.printVectorSpace(VSFile)
    VSFile.close()

    print("VectorSpace End")

    print("")
    print("Laplace Start")

    LaplaceFile = open("output/Porter/Porter_LaplaceFile.txt", "wt")
    MD.printLanguageModelLaplace(LaplaceFile)
    LaplaceFile.close()

    print("Laplace End")

    print("")
    print("JM Start")

    JMFile = open("output/Porter/Porter_JMFile.txt", "wt")
    MD.printLanguageModelJM(JMFile)
    JMFile.close()

    print("JM End")


if __name__ == '__main__':
    main()

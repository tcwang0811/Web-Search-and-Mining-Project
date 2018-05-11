from Models import Model


def main():
    # read data
    inputData = {}
    lenDict = {}
    totalTermAndDoc = []
    uniqueTerm = 27439  # 不在檔案中

    isHead = 1
    nowTerm = ""
    with open("data/toTest.txt") as file:
        for line in file:
            if isHead == 1:
                totalTermAndDoc = [int(i) for i in line[:-1].split(" ")]
                isHead = 0
            else:
                if line[0] != "\t":
                    temp = line[:-1].split(" ")
                    inputData[temp[0]] = [[int(i) for i in temp[1:]]]
                    nowTerm = temp[0]
                else:
                    temp = line[1:-1].split(" ")
                    toInput = [temp[0], int(temp[1])]  # 0: doc ID, 1: appear times
                    inputData[nowTerm].append(toInput)

                    if toInput[0] not in lenDict:
                        lenDict[toInput[0]] = int(toInput[1])
                    else:
                        lenDict[toInput[0]] += int(toInput[1])

    # read ID map
    ID_mapping = []
    with open("data/ID/ID_toTest.txt") as file:
        for line in file:
            temp = line[:-1].split(" ")
            ID_mapping.append(temp)

    # query not yet
    query = [["blood", "believe", "me"], ["permanent", "surface"]]

    # declare model
    MD = Model(inputData, lenDict, totalTermAndDoc, uniqueTerm, ID_mapping)
    MD.printVectorSpace(query)


if __name__ == '__main__':
    main()

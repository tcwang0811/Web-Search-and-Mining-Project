import math
import numpy as np
from PorterStemmer import PorterStemmer


class Model:
    # self.__index = file index
    # self.__lenDict = length of every document
    # self.__totalTermAndDoc = total number of term and number of document
    # self.__uniqueTerm = number of unique term
    # self.__ID = internal and external ID mapping
    # self.__query = query number and content
    # self.__stemmer = stemmer

    def __init__(self, pathData, pathID, pathQuery, uniqueT, isStemming):
        self.__uniqueTerm = uniqueT
        self.__buildData(pathData)
        self.__buildID(pathID)
        if isStemming:
            self.__stemmer = PorterStemmer()
        self.__buildQuery(pathQuery, isStemming)


    def __buildData(self, pathData):
        # read data
        self.__index = {}
        self.__lenDict = {}
        self.__totalTermAndDoc = []

        isHead = 1
        nowTerm = ""
        with open(pathData) as file:
            for line in file:
                if isHead == 1:
                    self.__totalTermAndDoc = [int(i) for i in line[:-1].split(" ")]
                    isHead = 0
                else:
                    if line[0] != "\t":
                        temp = line[:-1].split(" ")
                        self.__index[temp[0]] = [[int(i) for i in temp[1:]]]
                        nowTerm = temp[0]
                    else:
                        temp = line[1:-1].split(" ")
                        toInput = [temp[0], int(temp[1])]  # 0: doc ID, 1: appear times
                        self.__index[nowTerm].append(toInput)

                        if toInput[0] not in self.__lenDict:
                            self.__lenDict[toInput[0]] = int(toInput[1])
                        else:
                            self.__lenDict[toInput[0]] += int(toInput[1])

    def __buildID(self, pathID):
        # read ID map
        self.__ID = []
        with open(pathID) as file:
            for line in file:
                temp = line[:-1].split(" ")
                self.__ID.append(temp)

    @staticmethod
    def __clean(queryString):
        queryString = queryString.replace(",", "")
        # queryString = queryString.replace("\s+", " ")
        queryString = queryString.lower()
        output = queryString.split(" ")

        return output

    def __stem(self, queryWordList):
        return [self.__stemmer.stem(word, 0, len(word) - 1) for word in queryWordList]

    def __buildQuery(self, pathQuery, isStemming):
        self.__query = []
        temp = [0] * 2
        with open(pathQuery) as file:
            for line in file:
                if line[:5] == "<num>":
                    first = line.find(":") + 2
                    temp[0] = int(line[first:first + 3])
                elif line[:7] == "<title>":
                    first = line.find(">") + 2
                    if line.find(" \n") == -1:
                        last = -1
                    else:
                        last = -2
                    temp[1] = line[first:last]
                    self.__query.append(temp.copy())

        for i in range(len(self.__query)):
            self.__query[i][1] = self.__clean(self.__query[i][1])
            if isStemming:
                self.__query[i][1] = self.__stem(self.__query[i][1])

    def __VectorSpace(self, query):
        # Vector space model
        outputVS = {}
        listVS = []

        template = [0] * len(query)
        averageL = self.__totalTermAndDoc[0] / self.__totalTermAndDoc[1]

        # suppose every term in query appears only once
        queryTFIDF = template.copy()

        for i in range(len(query)):
            temp = self.__index[query[i]]
            nKey = temp[0][1] + 1
            for j in range(1, nKey):
                docID = temp[j][0]
                tfOfDoc = temp[j][1]
                if docID not in outputVS:
                    outputVS[docID] = template.copy()
                OkapiTF = tfOfDoc / (tfOfDoc + 0.5 + 1.5 * self.__lenDict[docID] / averageL)
                outputVS[docID][i] = OkapiTF * math.log(self.__totalTermAndDoc[1] / (nKey - 1))

            queryTFIDF[i] = 1 / (1 + 0.5 + 1.5 * len(query) / averageL) * math.log(
                self.__totalTermAndDoc[1] / (nKey - 1))

        for key, value in outputVS.items():
            temp = [key, np.dot(value, queryTFIDF)]
            listVS.append(temp)

        listVS.sort(key=lambda x: x[1], reverse=True)

        return listVS

    def __LanguageModelLaplace(self, query):
        outputLMLaplace = {}
        listLMLaplace = []

        for i in range(len(query)):
            temp = self.__index[query[i]]
            nKey = temp[0][1] + 1
            for j in range(1, nKey):
                docID = temp[j][0]
                tfOfDoc = temp[j][1]
                if docID not in outputLMLaplace:
                    outputLMLaplace[docID] = 0
                outputLMLaplace[docID] += math.log((tfOfDoc + 1) / (self.__lenDict[docID] + self.__uniqueTerm))

        for key, value in outputLMLaplace.items():
            temp = [key, value]
            listLMLaplace.append(temp)

        listLMLaplace.sort(key=lambda x: x[1], reverse=True)

        return listLMLaplace

    def __LanguageModelJM(self, query):
        outputLMJM = {}
        listLMJM = []

        # Let template be the P(|C)
        template = [0] * len(query)

        for i in range(len(query)):
            template[i] = (self.__index[query[i]][0][0] / self.__totalTermAndDoc[0]) * 0.8

        for i in range(len(query)):
            temp = self.__index[query[i]]
            nKey = temp[0][1] + 1
            for j in range(1, nKey):
                docID = temp[j][0]
                tfOfDoc = temp[j][1]
                if docID not in outputLMJM:
                    outputLMJM[docID] = template.copy()
                outputLMJM[docID][i] += (tfOfDoc / self.__lenDict[docID]) * 0.2

        for key in outputLMJM.keys():
            for j in range(len(query)):
                outputLMJM[key][j] = math.log(outputLMJM[key][j])
            outputLMJM[key] = sum(outputLMJM[key])

        for key, value in outputLMJM.items():
            temp = [key, value]
            listLMJM.append(temp)

        listLMJM.sort(key=lambda x: x[1], reverse=True)

        return listLMJM

    def printVectorSpace(self):
        for subQuery in range(len(self.__query)):
            VSList = self.__VectorSpace(self.__query[subQuery][1])

            if len(VSList) < 1000:
                    printIndex = len(VSList)
            else:
                printIndex = 1000

            for doc in range(1, printIndex + 1):
                print("%d Q0 %s %d %f Exp" % (self.__query[subQuery][0], self.__ID[int(VSList[doc - 1][0]) - 1][1], doc, VSList[doc - 1][1]))

            print("\n")

    def printLanguageModelLaplace(self):
        for subQuery in range(len(self.__query)):
            LaplaceList = self.__LanguageModelLaplace(self.__query[subQuery][1])

            if len(LaplaceList) < 1000:
                printIndex = len(LaplaceList)
            else:
                printIndex = 1000

            for doc in range(1, printIndex + 1):
                print("%d Q0 %s %d %f Exp" % (self.__query[subQuery][0], self.__ID[int(LaplaceList[doc - 1][0]) - 1][1], doc, LaplaceList[doc - 1][1]))

            print("\n")

    def printLanguageModelJM(self):
        for subQuery in range(len(self.__query)):
            JMList = self.__LanguageModelJM(self.__query[subQuery][1])

            if len(JMList) < 1000:
                printIndex = len(JMList)
            else:
                printIndex = 1000

            for doc in range(1, printIndex + 1):
                print("%d Q0 %s %d %f Exp" % (self.__query[subQuery][0], self.__ID[int(JMList[doc - 1][0]) - 1][1], doc, JMList[doc - 1][1]))

            print("\n")

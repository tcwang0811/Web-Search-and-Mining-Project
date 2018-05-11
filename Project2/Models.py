import math
import numpy as np


class Model:

    def __init__(self, inputData, lengthDict, termAndDoc, uniqueT, ID_mapping):
        self.__index = inputData
        self.__lenDict = lengthDict
        self.__totalTermAndDoc = termAndDoc
        self.__uniqueTerm = uniqueT
        self.__ID = ID_mapping

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

            queryTFIDF[i] = 1 / (1 + 0.5 + 1.5 * len(query) / averageL) * math.log(self.__totalTermAndDoc[1] / (nKey - 1))

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

    def printVectorSpace(self, query):
        for subQuery in range(len(query)):
            VSList = self.__VectorSpace(query[subQuery])

            if len(VSList) < 1000:
                printIndex = len(VSList)
            else:
                printIndex = 1000

            for doc in range(1, printIndex + 1):
                print("%d Q0 %s %d %f Exp" % (subQuery+1, self.__ID[int(VSList[doc - 1][0]) - 1][1], doc, VSList[doc - 1][1]))

            print("\n")

    def printLanguageModelLaplace(self, query):
        for subQuery in range(len(query)):
            LaplaceList = self.__LanguageModelLaplace(query[subQuery])

            if len(LaplaceList) < 1000:
                printIndex = len(LaplaceList)
            else:
                printIndex = 1000

            for doc in range(1, printIndex + 1):
                print("%d Q0 %s %d %f Exp" % (subQuery+1, self.__ID[int(LaplaceList[doc - 1][0]) - 1][1], doc, LaplaceList[doc - 1][1]))

            print("\n")

    def printLanguageModelJM(self, query):
        for subQuery in range(len(query)):
            JMList = self.__LanguageModelJM(query[subQuery])

            if len(JMList) < 1000:
                printIndex = len(JMList)
            else:
                printIndex = 1000

            for doc in range(1, printIndex + 1):
                print("%d Q0 %s %d %f Exp" % (subQuery+1, self.__ID[int(JMList[doc - 1][0]) - 1][1], doc, JMList[doc - 1][1]))

            print("\n")

    @staticmethod
    def queryPreprocess(query):



        return
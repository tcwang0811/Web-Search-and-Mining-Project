import math
import numpy as np


class Model:
    index = {}
    lenDict = {}
    totalTermAndDoc = []
    ID = []

    def __init__(self, inputData, lengthDict, termAndDoc, uniqueT, ID_mapping):
        self.index = inputData
        self.lenDict = lengthDict
        self.totalTermAndDoc = termAndDoc
        self.uniqueTerm = uniqueT
        self.ID = ID_mapping

    def VectorSpace(self, query):
        # Vector space model
        outputVS = {}
        listVS = []

        template = [0] * len(query)
        averageL = self.totalTermAndDoc[0] / self.totalTermAndDoc[1]

        # suppose every term in query appears only once
        queryTFIDF = template.copy()

        for i in range(len(query)):
            temp = self.index[query[i]]
            nKey = temp[0][1] + 1
            for j in range(1, nKey):
                docID = temp[j][0]
                tfOfDoc = temp[j][1]
                if docID not in outputVS:
                    outputVS[docID] = template.copy()
                OkapiTF = tfOfDoc / (tfOfDoc + 0.5 + 1.5 * self.lenDict[docID] / averageL)
                outputVS[docID][i] = OkapiTF * math.log(self.totalTermAndDoc[1] / (nKey - 1))

            queryTFIDF[i] = 1 / (1 + 0.5 + 1.5 * len(query) / averageL) * math.log(self.totalTermAndDoc[1] / (nKey - 1))

        for key, value in outputVS.items():
            temp = [key, np.dot(value, queryTFIDF)]
            listVS.append(temp)

        listVS.sort(key=lambda x: x[1], reverse=True)

        return listVS

    def LanguageModelLaplace(self, query):
        outputLMLaplace = {}
        listLMLaplace = []

        for i in range(len(query)):
            temp = self.index[query[i]]
            nKey = temp[0][1] + 1
            for j in range(1, nKey):
                docID = temp[j][0]
                tfOfDoc = temp[j][1]
                if docID not in outputLMLaplace:
                    outputLMLaplace[docID] = 0
                outputLMLaplace[docID] += math.log((tfOfDoc + 1) / (self.lenDict[docID] + self.uniqueTerm))

        for key, value in outputLMLaplace.items():
            temp = [key, value]
            listLMLaplace.append(temp)

        listLMLaplace.sort(key=lambda x: x[1], reverse=True)

        return listLMLaplace

    def LanguageModelJM(self, query):
        outputLMJM = {}
        listLMJM = []

        # Let template be the P(|C)
        template = [0] * len(query)

        for i in range(len(query)):
            template[i] = (self.index[query[i]][0][0] / self.totalTermAndDoc[0]) * 0.8

        for i in range(len(query)):
            temp = self.index[query[i]]
            nKey = temp[0][1] + 1
            for j in range(1, nKey):
                docID = temp[j][0]
                tfOfDoc = temp[j][1]
                if docID not in outputLMJM:
                    outputLMJM[docID] = template.copy()
                outputLMJM[docID][i] += (tfOfDoc / self.lenDict[docID]) * 0.2

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
            VSList = self.VectorSpace(query[subQuery])

            if len(VSList) < 1000:
                printIndex = len(VSList)
            else:
                printIndex = 1000

            for doc in range(1, printIndex + 1):
                print("%d Q0 %s %d %f Exp" % (subQuery+1, self.ID[int(VSList[doc - 1][0]) - 1][1], doc, VSList[doc - 1][1]))

            print("\n")


import math
import numpy as np
from PorterStemmer import PorterStemmer


class Model:
    # self.__index = file index (only term appears in query in index)
    # self.__lenDict = length of every document (all)
    # self.__totalTermAndDoc = total number of term and number of document
    # self.__uniqueTerm = number of unique term
    # self.__ID = internal and external ID mapping
    # self.__query = query number and content
    # self.__querySet = a set to test if a term in the query
    # self.__stemmer = stemmer

    def __init__(self, pathData, pathID, pathQuery, uniqueT, isStemming):
        """

        :param pathData: string of pathData
        :param pathID: string of pathID
        :param pathQuery: string of pathQuery
        :param uniqueT: integer of unique term
        :param isStemming: True if using stemming, False if not using stemming
        """
        self.__buildID(pathID)
        if isStemming:
            self.__stemmer = PorterStemmer()
        self.__buildQuery(pathQuery, isStemming)
        self.__uniqueTerm = uniqueT

        self.__buildData(pathData)


    def __buildData(self, pathData):
        # read data
        self.__index = {}
        self.__lenDict = {}
        self.__totalTermAndDoc = []

        isHead = 1
        isInSet = 0
        nowTerm = ""
        with open(pathData, encoding = 'utf8') as file:
            for line in file:
                if isHead == 1:
                    # Top of Data
                    # Add total number of term and number of document
                    self.__totalTermAndDoc = [int(i) for i in line[:-1].split(" ")]
                    isHead = 0
                else:
                    if line[0] != "\t":
                        # Beginning of term
                        temp = line[:-1].split(" ")
                        # is term appear in query?
                        if temp[0] in self.__querySet:
                            # Add term to key and 0 position is total appear time and appear document
                            self.__index[temp[0]] = [[int(i) for i in temp[1:]]]
                            nowTerm = temp[0]
                            isInSet = 1
                        else:
                            isInSet = 0
                    else:
                        # not beginning of term
                        temp = line[1:-1].split(" ")
                        toInput = [temp[0], int(temp[1])]  # 0: doc ID, 1: appear times

                        # is term appear in query?
                        if isInSet == 1:
                            # Add doc ID and appear time
                            self.__index[nowTerm].append(toInput)

                        # Add doc length
                        if toInput[0] not in self.__lenDict:
                            self.__lenDict[toInput[0]] = int(toInput[1])
                        else:
                            self.__lenDict[toInput[0]] += int(toInput[1])

    def __buildID(self, pathID):
        # read ID map
        self.__ID = []
        with open(pathID, encoding = 'utf8') as file:
            for line in file:
                # Add internal ID and external ID to self.__ID
                temp = line[:-1].split(" ")
                self.__ID.append(temp)

    @staticmethod
    def __clean(queryString):
        """

        :param queryString: a query string
        :return: a list of cleaned query
        """
        queryString = queryString.replace(",", "")
        # queryString = queryString.replace("\s+", " ")
        queryString = queryString.replace("'", "")
        queryString = queryString.replace(".", "")
        queryString = queryString.lower()
        output = queryString.split(" ")

        return output

    def __stem(self, queryWordList):
        """

        :param queryWordList: a list of cleaned query
        :return: a list of stemmed query
        """
        return [self.__stemmer.stem(word, 0, len(word) - 1) for word in queryWordList]

    def __buildQuery(self, pathQuery, isStemming):
        self.__query = []
        self.__querySet = []
        temp = [0] * 2

        with open(pathQuery, encoding = 'utf8') as file:
            for line in file:
                if line[:5] == "<num>":
                    # update query number
                    first = line.find(":") + 2
                    temp[0] = int(line[first:first + 3])
                elif line[:7] == "<title>":
                    # update query
                    first = line.find(">") + 2
                    if line.find(" \n") == -1:
                        last = -1
                    else:
                        last = -2
                    temp[1] = line[first:last]
                    # Add query number and query to self.__query
                    self.__query.append(temp.copy())

        # clean and stem
        for i in range(len(self.__query)):
            self.__query[i][1] = self.__clean(self.__query[i][1])
            if isStemming:
                self.__query[i][1] = self.__stem(self.__query[i][1])
            # add every query term to set
            self.__querySet.extend(self.__query[i][1])

        # transfer to set
        self.__querySet = set(self.__querySet)

    def __VectorSpace(self, query):
        """

        :param query: a list of query term
        :return: a list of internal Doc ID and score which is sorted from high to low
        """
        # Vector space model
        outputVS = {}
        listVS = []

        # Suppose every term in query appears only once
        template = [0] * len(query)
        averageL = self.__totalTermAndDoc[0] / self.__totalTermAndDoc[1]

        queryTFIDF = template.copy()

        # Calculate TF-IDF of every term
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

        # to list and sort
        for key, value in outputVS.items():
            temp = [key, np.dot(value, queryTFIDF)]
            listVS.append(temp)

        listVS.sort(key=lambda x: x[1], reverse=True)

        return listVS

    def __LanguageModelLaplace(self, query):
        """

        :param query: a list of query term
        :return: a list of internal Doc ID and score which is sorted from high to low
        """
        outputLMLaplace = {}
        listLMLaplace = []

        # Calculate every probability of document with Laplace smoothing
        for i in range(len(query)):
            temp = self.__index[query[i]]
            nKey = temp[0][1] + 1
            for j in range(1, nKey):
                docID = temp[j][0]
                tfOfDoc = temp[j][1]
                docLength = self.__lenDict[docID]
                if docID not in outputLMLaplace:
                    outputLMLaplace[docID] = [math.log(1/(docLength + self.__uniqueTerm))] * len(query)
                outputLMLaplace[docID][i] = math.log((tfOfDoc + 1) / (docLength + self.__uniqueTerm))

        # sum
        for key in outputLMLaplace.keys():
            outputLMLaplace[key]=sum(outputLMLaplace[key])

        # to list and sort
        for key, value in outputLMLaplace.items():
            temp = [key, value]
            listLMLaplace.append(temp)

        listLMLaplace.sort(key=lambda x: x[1], reverse=True)

        return listLMLaplace

    def __LanguageModelJM(self, query):
        """

        :param query: a list of query term
        :return: a list of internal Doc ID and score which is sorted from high to low
        """
        outputLMJM = {}
        listLMJM = []

        # Let template be the probability of corpus
        template = [0] * len(query)

        for i in range(len(query)):
            template[i] = (self.__index[query[i]][0][0] / self.__totalTermAndDoc[0]) * 0.8

        # Calculate every probability of document and add to its term
        for i in range(len(query)):
            temp = self.__index[query[i]]
            nKey = temp[0][1] + 1
            for j in range(1, nKey):
                docID = temp[j][0]
                tfOfDoc = temp[j][1]
                if docID not in outputLMJM:
                    outputLMJM[docID] = template.copy()
                outputLMJM[docID][i] += (tfOfDoc / self.__lenDict[docID]) * 0.2

        # Take natural log and sum
        for key in outputLMJM.keys():
            for j in range(len(query)):
                outputLMJM[key][j] = math.log(outputLMJM[key][j])
            outputLMJM[key] = sum(outputLMJM[key])

        # to list and sort
        for key, value in outputLMJM.items():
            temp = [key, value]
            listLMJM.append(temp)

        listLMJM.sort(key=lambda x: x[1], reverse=True)

        return listLMJM

    def printVectorSpace(self, fileName):
        for subQuery in range(len(self.__query)):
            # for every query, calculate vector space rank
            VSList = self.__VectorSpace(self.__query[subQuery][1])

            # determine whether length of List longer than 1000
            if len(VSList) < 1000:
                    printIndex = len(VSList)
            else:
                printIndex = 1000

            # print outcome
            for doc in range(1, printIndex + 1):
                print("%d Q0 %s %d %f Exp" % (self.__query[subQuery][0], self.__ID[int(VSList[doc - 1][0]) - 1][1], doc, VSList[doc - 1][1]), file=fileName)

    def printLanguageModelLaplace(self, fileName):
        for subQuery in range(len(self.__query)):
            # for every query, calculate language model with Laplace smoothing rank
            LaplaceList = self.__LanguageModelLaplace(self.__query[subQuery][1])

            # determine whether length of List longer than 1000
            if len(LaplaceList) < 1000:
                printIndex = len(LaplaceList)
            else:
                printIndex = 1000

            # print outcome
            for doc in range(1, printIndex + 1):
                print("%d Q0 %s %d %f Exp" % (self.__query[subQuery][0], self.__ID[int(LaplaceList[doc - 1][0]) - 1][1], doc, LaplaceList[doc - 1][1]), file=fileName)

    def printLanguageModelJM(self, fileName):
        for subQuery in range(len(self.__query)):
            # for every query, calculate language model with JM smoothing rank
            JMList = self.__LanguageModelJM(self.__query[subQuery][1])

            # determine whether length of List longer than 1000
            if len(JMList) < 1000:
                printIndex = len(JMList)
            else:
                printIndex = 1000

            # print outcome
            for doc in range(1, printIndex + 1):
                print("%d Q0 %s %d %f Exp" % (self.__query[subQuery][0], self.__ID[int(JMList[doc - 1][0]) - 1][1], doc, JMList[doc - 1][1]), file=fileName)

    ## try cosine ##
    def __VectorSpaceCosine(self, query, normSquareDict):
        """

        :param query: a list of query term
        :return: a list of internal Doc ID and score which is sorted from high to low
        """
        # Vector space model
        outputVS = {}
        listVS = []

        # Suppose every term in query appears only once
        template = [0] * len(query)
        averageL = self.__totalTermAndDoc[0] / self.__totalTermAndDoc[1]

        queryTFIDF = template.copy()

        # Calculate TF-IDF of every term
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

        # to list and sort
        for key, value in outputVS.items():
            docNorm = math.sqrt(normSquareDict[key])
            score = np.dot(value, queryTFIDF)/(np.linalg.norm(queryTFIDF) * docNorm)

            temp = [key, score]
            listVS.append(temp)

        listVS.sort(key=lambda x: x[1], reverse=True)

        return listVS

    def __buildNorm(self, pathData):
        normSquareDict = {}
        averageL = self.__totalTermAndDoc[0] / self.__totalTermAndDoc[1]

        isHead = 1
        nDoc = 0
        with open(pathData, encoding='utf8') as file:
            for line in file:
                if isHead == 1:
                    # Top of Data
                    isHead = 0
                else:
                    if line[0] != "\t":
                        # Beginning of term
                        temp = line[:-1].split(" ")
                        nDoc = int(temp[2])
                    else:
                        # not beginning of term
                        temp = line[1:-1].split(" ")
                        docID = temp[0]
                        tfOfDoc = int(temp[1])

                        OkapiTF = tfOfDoc / (tfOfDoc + 0.5 + 1.5 * self.__lenDict[docID] / averageL)
                        TFIDF = OkapiTF * math.log(self.__totalTermAndDoc[1] / nDoc)

                        # Add doc length
                        if docID not in normSquareDict:
                            normSquareDict[docID] = TFIDF ** 2
                        else:
                            normSquareDict[docID] += TFIDF ** 2

        return normSquareDict

    def printVectorSpaceCosine(self, fileName, pathData):

        normSquareDict = self.__buildNorm(pathData)

        for subQuery in range(len(self.__query)):
            # for every query, calculate vector space rank
            VSList = self.__VectorSpaceCosine(self.__query[subQuery][1], normSquareDict)

            # determine whether length of List longer than 1000
            if len(VSList) < 1000:
                    printIndex = len(VSList)
            else:
                printIndex = 1000

            # print outcome
            for doc in range(1, printIndex + 1):
                print("%d Q0 %s %d %f Exp" % (self.__query[subQuery][0], self.__ID[int(VSList[doc - 1][0]) - 1][1], doc, VSList[doc - 1][1]), file=fileName)
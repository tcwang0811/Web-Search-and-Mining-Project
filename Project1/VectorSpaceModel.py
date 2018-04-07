# VectorSpace
from Parser import Parser
import util
# tf-idf
import math

# ------------------------------------------------- #

# 1. calculate the tf or tf-idf
# input: collection of documents
# output: the vector of each document

# 2. return the similarity
# input: a query
# output: the similarity of each document

# ------------------------------------------------- #

class VectorSpaceModel:
    """ A algebraic model for representing text documents as vectors of identifiers.
    A document is represented as a vector. Each dimension of the vector corresponds to a
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    # Collection of document term vectors
    documentVectors = []

    # Mapping of vector index to keyword
    vectorKeywordIndex = []

    # IDF
    IDFVector = []

    # TF-TDF
    TFIDFVector = []

    # Tidies terms
    parser = None

    def __init__(self, documents=[]):
        self.documentVectors = []
        self.parser = Parser()
        if (len(documents) > 0):
            self.build(documents)

    def build(self, documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        self.documentVectors = [self.makeSimpleVector(document) for document in documents]
        self.IDFVector = self.makeIDFVector(documents)
        self.TFIDFVector = [self.makeTFIDFVector(TFV) for TFV in self.documentVectors]

        # print self.vectorKeywordIndex
        # print self.documentVectors

    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        # Mapped documents into a single word string
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)
        # Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex = {}
        offset = 0
        # Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word] = offset
            offset += 1
        return vectorIndex  # (keyword:position)

    def makeSimpleVector(self, wordString):
        """ @pre: unique(vectorIndex) """

        # Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        for word in wordList:
            vector[self.vectorKeywordIndex[word]] += 1  # Use simple Term Count Model
        return vector

    def makeIDFVector(self, documentList):

        outputVector = [0] * len(self.vectorKeywordIndex)

        keyVector = list(self.vectorKeywordIndex.keys())
        docNumber = len(documentList)  # should be 2048

        for i in range(docNumber):
            docTemp = self.parser.tokenise(documentList[i])
            docTemp = self.parser.removeStopWords(docTemp)
            uniqueDocTemp = util.removeDuplicates(docTemp)

            for key in uniqueDocTemp:
                outputVector[self.vectorKeywordIndex[key]] += 1 # DF

        for key in keyVector:
            outputVector[self.vectorKeywordIndex[key]] = math.log(docNumber/outputVector[self.vectorKeywordIndex[key]]) # IDF

        return outputVector

    def makeTFIDFVector(self, TFVector):

        outputVector = [0] * len(TFVector)

        for i in range(len(TFVector)):
            outputVector[i] = self.IDFVector[i]*TFVector[i]

        return outputVector

    def buildSimpleQueryVector(self, termList):
        """ convert query string into a TF vector """
        query = self.makeSimpleVector(" ".join(termList))
        return query

    def buildTFIDFQueryVector(self, termList):
        """ convert query string into a TF-IDF vector """
        TFQuery = self.buildSimpleQueryVector(termList)
        query = self.makeTFIDFVector(TFQuery)
        return query

    def searchTFWithCosine(self, searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildSimpleQueryVector(searchList)

        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        # ratings.sort(reverse=True)
        return ratings

    def searchTFIDFWithCosine(self, searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildTFIDFQueryVector(searchList)

        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.TFIDFVector]
        # ratings.sort(reverse=True)
        return ratings

    def searchTFWithEuclideanDist(self, searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildSimpleQueryVector(searchList)

        ratings = [util.EuclideanDist(queryVector, documentVector) for documentVector in self.documentVectors]
        # ratings.sort(reverse=True)
        return ratings

    def searchTFIDFWithEuclideanDist(self, searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildTFIDFQueryVector(searchList)

        ratings = [util.EuclideanDist(queryVector, documentVector) for documentVector in self.TFIDFVector]
        # ratings.sort(reverse=True)
        return ratings
# VectorSpace
from Parser import Parser
import util
# tf-idf
import math
import TF_IDF
from textblob import TextBlob as tb
# Calculate Similarity
import Similarity

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

        outputVector = self.vectorKeywordIndex

        keyVector = list(outputVector.keys())
        docNumber = len(documentList)  # should be 2048

        # Initialize
        for key in keyVector:
            outputVector[key] = 0

        for i in range(docNumber):
            docTemp = self.parser.tokenise(documentList[i])
            docTemp = self.parser.removeStopWords(docTemp)
            uniqueDocTemp = util.removeDuplicates(docTemp)

            for key in uniqueDocTemp:
                outputVector[key] += 1 # DF

        for key in keyVector:
            outputVector[key] = math.log(docNumber/outputVector[key]) # IDF

        return outputVector

    def makeTFIDFVector(self, TFVector):

        outputVector = [0] * len(TFVector)












    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        query = self.makeSimpleVector(" ".join(termList))
        return query

    def related(self, documentId):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in
                   self.documentVectors]
        # ratings.sort(reverse=True)
        return ratings

    def search(self, searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        # ratings.sort(reverse=True)
        return ratings

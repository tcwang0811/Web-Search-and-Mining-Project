from Parser import Parser
import util
import math

class VectorSpaceModel:
    """ A algebraic model for representing text documents as vectors of identifiers.
    A document is represented as a vector. Each dimension of the vector corresponds to a
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    # Collection of document term vectors
    TFVectors = []

    # Mapping of vector index to keyword
    vectorKeywordIndex = []

    # IDF
    IDFVector = []

    # TF-TDF
    TFIDFVector = []

    # Tidies terms
    parser = None

    def __init__(self, documents=[]):
        self.TFVectors = []
        self.parser = Parser()
        if (len(documents) > 0):
            self.build(documents)

    def build(self, documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        self.TFVectors = [self.makeTFVector(document) for document in documents]
        self.IDFVector = self.makeIDFVector(documents)
        self.TFIDFVectors = [self.makeTFIDFVector(TFV) for TFV in self.TFVectors]

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

    def makeTFVector(self, wordString):
        """ @pre: unique(vectorIndex) """

        # Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)

        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)

        for word in wordList:
            vector[self.vectorKeywordIndex[word]] += 1

        return vector

    def makeIDFVector(self, documentList):

        outputVector = [0] * len(self.vectorKeywordIndex)

        docNumber = len(documentList)

        for doc in documentList:
            docTemp = self.parser.tokenise(doc)
            docTemp = self.parser.removeStopWords(docTemp)
            uniqueDocTemp = util.removeDuplicates(docTemp)

            for key in uniqueDocTemp:
                outputVector[self.vectorKeywordIndex[key]] += 1 # DF

        for i in range(len(outputVector)):
            outputVector[i] = math.log(docNumber/outputVector[i]) # IDF

        return outputVector

    def makeTFIDFVector(self, TFVector):
        outputVector = [0] * len(TFVector)

        for i in range(len(TFVector)):
            outputVector[i] = self.IDFVector[i]*TFVector[i]

        return outputVector

    def buildTFQueryVector(self, termList):
        """ convert query string into a TF vector """
        query = self.makeTFVector(" ".join(termList))
        return query

    def buildTFIDFQueryVector(self, termList):
        """ convert query string into a TF-IDF vector """
        TFQuery = self.buildTFQueryVector(termList)
        query = self.makeTFIDFVector(TFQuery)
        return query

    def searchTFWithCosine(self, searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildTFQueryVector(searchList)

        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.TFVectors]

        return ratings

    def searchTFIDFWithCosine(self, searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildTFIDFQueryVector(searchList)

        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.TFIDFVectors]

        return ratings

    def searchTFWithEuclideanDist(self, searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildTFQueryVector(searchList)

        ratings = [util.EuclideanDist(queryVector, documentVector) for documentVector in self.TFVectors]

        return ratings

    def searchTFIDFWithEuclideanDist(self, searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildTFIDFQueryVector(searchList)

        ratings = [util.EuclideanDist(queryVector, documentVector) for documentVector in self.TFIDFVectors]

        return ratings

    def searchRelevantFeedback(self, QueryFeedback):
        # first element is query, second element is relevant feedback

        query = self.buildTFIDFQueryVector(QueryFeedback[0])
        feedback = self.buildTFIDFQueryVector(QueryFeedback[1])

        NewQuery = [0]*len(query)

        for i in range(len(query)):
            NewQuery[i] = 1 * query[i] + 0.5 * feedback[i]

        ratings = [util.cosine(NewQuery, documentVector) for documentVector in self.TFIDFVectors]

        return ratings
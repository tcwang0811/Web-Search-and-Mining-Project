import nltk
from Parser import Parser

def main():
    # open and store document
    path = "documents/"
    file = "187855.product"

    documents = []
    documentsNo = []

    file187855 = open(path + file, 'r')
    documents.append(file187855.read())
    documentsNo.append(file)
    file187855.close()
    ################################

    text = nltk.word_tokenize(documents[0])
    a = nltk.pos_tag(text)

    output = ""

    print(a)

    for i in range(len(a)):
        if a[i][1][0] == "N" or a[i][1][0] == "V":
            if output != "":
                output += " "
            output += a[i][0]

    print(output)









if __name__ == '__main__':
    main()
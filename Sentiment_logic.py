import nltk
from nltk.classify import NaiveBayesClassifier


def format_sentence(sent):
    return ({word: 'token' for word in nltk.word_tokenize(sent)})


def start_sentiment_analysis(test_sentence):
    pos = []
    with open("./pos_sent.txt") as file:  # open positive sentences
        for sent in file:
            pos.append([format_sentence(sent), 'Sentiment = positive'])

    neg = []
    with open("./neg_sent.txt") as file:  # open negative sentences
        for sent in file:
            neg.append([format_sentence(sent), 'Sentiment = negative'])


    # next, split labeled data into the training
    training = pos + neg

    classifier = NaiveBayesClassifier.train(training)
    output = classifier.classify(format_sentence(test_sentence))
    return output  # will return if pos if sentence is positive vice-versa


input_sentence = input('Enter Sentence please_> ')  # example = 'donald trump is a good man'
output = start_sentiment_analysis(input_sentence)
print(output)
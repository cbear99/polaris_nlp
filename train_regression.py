#!/usr/bin/env python3.7

import argparse
import csv
import string
import ssl
import nltk
import pandas
from joblib import dump
from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def clean_text_for_vectorizer(row: List, stopwords: List) -> str:
    # Remove phrases that are helper words in text (eg REASON AND CONTEXT FOR SIGNAL)
    text = row['notes']
    helper_free_text = text.replace('REASON AND CONTEXT FOR SIGNAL', '').replace('POTENTIAL VICTIM', ''). \
        replace('POTENTIAL TRAFFICKER', '').replace('SITUATION INFORMATION', '').replace('SIGNALER', ''). \
        replace('SUSPICIOUS BUSINESS', '')
    # Remove unnecessary characters
    no_punctuation = ""
    for char in helper_free_text:
        if char not in string.punctuation:
            no_punctuation += char
        else:
            no_punctuation += ' '
    # Normalize to lowercase
    lower = no_punctuation.lower()
    # Tokenize into words
    words = lower.split()
    lemmatizer = nltk.WordNetLemmatizer()
    # Reduce common words (and n which gets left over from the new line punctuation parsing)
    reduced_words = []
    for word in words:
        if word not in stopwords and word != 'n':
            # use nltk's Lemmatizer to get the root of a given word (eg cats -> cat, running -> run)
            lemma = lemmatizer.lemmatize(word)
            reduced_words.append(lemma)
    return reduced_words


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Ranking for CSV')

    parser.add_argument("input_file", help="Path to a CSV File to generate a ranking for")
    # NOTE: Had to download the nltk stopwords manually rather than use nltk.corpus.stopwords("english")
    parser.add_argument("stopwords", help="Path to stopwords file")

    args = parser.parse_args()
    print("Beginning CSV Processing")

    # NOTE: To enable nltk interface, use hack from StackOverflow
    # https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('wordnet')
    nltk.download('omw-1.4')

    with open(args.input_file, newline='') as input_file, \
            open(args.stopwords, newline='') as stopwords:
        stopwords_array = stopwords.readlines()
        reader = csv.DictReader(input_file)
        raw_data = []
        for row in reader:
            raw_data.append(row)
        # Use Bag of Words to create document-term matrix
        # NOTE: In production would also consider using a different vectorizer, such as
        # using relative frequency or n-grams, but felt beyond the scope of this assignment
        vectorizer = CountVectorizer(analyzer=lambda x: clean_text_for_vectorizer(x, stopwords_array))

        X = vectorizer.fit_transform(raw_data).toarray()
        dataset = pandas.DataFrame(raw_data)
        # This assumes the training dataset has values for whether a case resulted in actionable intelligence
        # for the Financial Intelligence Unit
        y = dataset.iloc[:, 16].values

        print("Training a Multinomial Naive Bayes Classifier on 25 percent of the dataset")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, train_size=0.25, random_state=0)
        mnb = MultinomialNB()
        classifier = mnb.fit(X_train, y_train)
        print("Testing on the other 75 Percent of the Dataset ")
        y_pred = mnb.predict(X_test)
        print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
        dump(mnb, 'multinomial_naive_bayes.joblib')

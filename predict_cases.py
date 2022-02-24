#!/usr/bin/env python3.7

import argparse
import csv
import pandas
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from train_regression import clean_text_for_vectorizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Ranking for CSV')

    parser.add_argument("input_file", help="Path to a CSV File to generate a ranking for")
    parser.add_argument("output_file", help="Name of the file to output ranking")
    parser.add_argument("stopwords", help="Path to stopwords file")

    args = parser.parse_args()
    print("Beginning CSV Processing")

    with open(args.input_file, newline='') as input_file, \
            open(args.output_file, 'w') as output_file, \
            open(args.stopwords, newline='') as stopwords:
        stopwords_array = stopwords.readlines()
        reader = csv.DictReader(input_file)
        writer = csv.DictWriter(output_file, fieldnames=['ID', 'Ranking'], lineterminator='\n')
        raw_data = []
        for row in reader:
            raw_data.append(row)
        # Use Bag of Words to create document-term matrix
        # NOTE: In production would also consider using a different vectorizer, such as
        # using relative frequency or n-grams, but felt beyond the scope of this assignment
        vectorizer = CountVectorizer(analyzer=lambda x: clean_text_for_vectorizer(x, stopwords_array))

        X = vectorizer.fit_transform(raw_data).toarray()
        mnb = load('multinomial_naive_bayes.joblib')
        probabilities = mnb.predict_proba(X)
        # Generate Mapping from ID to Probability
        dataframe = pandas.DataFrame(raw_data)
        mapping = {}
        for index, prob in enumerate(probabilities):
            # We want the probability a case is 1, (meaning the probability this is a case has actionable intelligence)
            mapping[dataframe.iloc[index]['id']] = prob[1]
        writer.writeheader()
        # Write to output file
        for index, case_id in enumerate(sorted(mapping.items(), key=lambda x: x[1], reverse=True)):
            writer.writerow({"ID": case_id[0], "Ranking": index})
        print("Finished writing to output file")






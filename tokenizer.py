import spacy
from spacy.lang.en.examples import sentences
import pandas as pd
from clean_csv import load_dataset

# Get dataset
dataset = load_dataset("email_spam.csv")

# Loading english pipeline
nlp = spacy.load('en_core_web_sm')

# Get vector representation of sentences
dataset['vector'] = dataset['messages'].apply(lambda x: nlp(x).vector)

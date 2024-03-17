import spacy
from spacy.lang.en.examples import sentences
import pandas as pd

# Loading english pipeline
nlp = spacy.load('en_core_web_sm')

# Load the haram dataset
data = pd.read_csv("email_spam.csv", encoding="utf-8", encoding_errors="replace")

# Merge data in one table
for message in data:
    message.column

print(f)
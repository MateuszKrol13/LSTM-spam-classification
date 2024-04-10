import pandas as pd
import string
from nltk.corpus import stopwords
import spacy
import numpy as np

def remove_punctuations(text):
    punctuations_list = string.punctuation
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

def remove_stopwords(text):
    stop_words = stopwords.words('english')
    imp_words = []

    # Storing the important words
    for word in str(text).split():
        word = word.lower()
        if word not in stop_words:
            imp_words.append(word)

    output = " ".join(imp_words)
    return output


def load_dataset(datasetPath):
    usecols = ["class", "message", "c1", "c2", "c3"]

    # Load dataset
    data = pd.read_csv(datasetPath, encoding="utf-8", encoding_errors="replace", names=usecols)
    data = data.iloc[1:] # get rid of first row

    # TODO: Rewrite this, this is ugly
    # join messages, as they are split in excel
    messages = data[data.columns[1:]]
    tmp = messages["message"].astype(str) + messages['c1'].astype(str) + messages['c2'].astype(str) + messages['c3'].astype(str)

    # Create main dataset
    dataset = pd.DataFrame()
    dataset["messages"] = tmp.apply(lambda x: remove_punctuations(x))
    dataset["messages"] = dataset["messages"].apply(lambda x: remove_stopwords(x))
    dataset["class"] = pd.factorize(data["class"])[0]

    return dataset

def prepare_input_data(datasetPath):
    # Get dataset
    dataset = load_dataset("email_spam.csv")

    # Split and balance the dataset
    spam = dataset[dataset['class'] == 1]
    messages = dataset[dataset['class'] == 0]
    messages = messages.sample(n=len(spam), random_state=42)

    # Merge dataset back
    dataset = pd.concat([spam, messages])
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    # Loading english pipeline
    nlp = spacy.load('en_core_web_sm')

    # Get vector representation of sentences
    dataset['vector'] = dataset['messages'].apply(lambda x: nlp(x).vector)

    return dataset

def pandas_to_numpy(pandas_array):
    vec = pandas_array.to_numpy()
    npy_arry = np.zeros((len(vec), len(vec[0])), dtype=np.float32)
    for i in range(len(vec)):
        npy_arry[i, :] = vec[i]

    return npy_arry
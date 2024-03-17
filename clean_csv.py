import pandas as pd

usecols = ["class", "message", "c1", "c2", "c3"]

# Load dataset
data = pd.read_csv("email_spam.csv", encoding="utf-8", encoding_errors="replace", names=usecols)
data = data.iloc[1:] # get rid of first row

# Split origian dataset
classes = pd.factorize(data["class"])[0]
messages = data[data.columns[1:]]

# TODO: Rewrite this, this is ugly
# join split messages
messages = messages["message"] + messages['c1'] + messages['c2'] + messages['c3']

print("end")
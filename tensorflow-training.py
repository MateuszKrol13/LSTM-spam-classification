from clean_csv import prepare_input_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from clean_csv import pandas_to_numpy

dataset = prepare_input_data("email_spam.csv")

input_vectors = pandas_to_numpy(dataset['vector'])

# Train vs Other
train_X, model_X, train_Y, model_Y = train_test_split(input_vectors, np.asarray(dataset['class']),
                                                      test_size=0.3, random_state=42)

# Test vs Validate
validate_X, test_X, validate_Y, test_Y = train_test_split(model_X, model_Y,
                                                          test_size=0.5, random_state=42)

# No padding needed, sequences are of equal length
# len(max(test_X, key=len) == len(min(test_X, key=len)
# len(max(train_X, key=len) == len(min(train_X, key=len)
input_length = len(train_X[0])

# Build the model
model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.LSTM(64))
model.add(tf.keras.layers.Input(input_length))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Print the model summary
model.summary()

# Loss, metrics, optimizer
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'],
              optimizer='adam')

# Callbacks
es = EarlyStopping(patience=3,
                   monitor='val_accuracy',
                   restore_best_weights=True)

lr = ReduceLROnPlateau(patience=2,
                       monitor='val_loss',
                       factor=0.5,
                       verbose=0)

# Train the model
history = model.fit(train_X, train_Y,
                    validation_data=(validate_X, validate_Y),
                    epochs=20,
                    batch_size=4,
                    callbacks=[lr, es]
                    )

# Predict
predictions = model.predict(test_X)
pred = predictions.round(decimals=0)
result = confusion_matrix(test_Y, pred)
print("=== confusion matrix ===")
print(result)

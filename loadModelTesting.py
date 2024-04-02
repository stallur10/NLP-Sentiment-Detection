import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import numpy as np
import requests
from bs4 import BeautifulSoup
from html.parser import HTMLParser

TF_ENABLE_ONEDNN_OPTS=0
max_words = 10000  
max_sequence_length = 100  

user_input = "You are very good!!, keep it up"
tokenizer = Tokenizer(num_words=max_words)
user_sequences = tokenizer.texts_to_sequences([user_input])
user_padded = tf.keras.preprocessing.sequence.pad_sequences(user_sequences, maxlen=max_sequence_length)

from keras.models import load_model
# loaded_model = load_model('savedModel')  

# class_mapping={
# -1:'Negative',
# 0:'Neutral',
# 1:'Positive'}
# # Make predictions on user input
# user_predictions = loaded_model.predict(user_padded)
# # Convert predictions to class labels (assuming it's a classification task)
# user_pred_classes = np.argmax(user_predictions, axis=1)
# # Print the predicted class
# print(f'Predicted Class: {class_mapping[user_pred_classes[0]]}')
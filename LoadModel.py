from keras.models import load_model
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import numpy as np
import requests
from bs4 import BeautifulSoup
from html.parser import HTMLParser

# response = requests.get("https://www.reddit.com/r/stocks/new/")

# if response.status_code == 200:
        
#         soup = BeautifulSoup(response.content, "html.parser")

        # paragraph = soup.find(id = "t3_1b898d8-post-rtjson-content")

#         print(paragraph.get_text())

# else:
        
#         print("Request Failed")
paragraph = "Hi I am so happy I love everything this is great."
        
max_words = 10000  
max_sequence_length = 100  
tokenizer = Tokenizer(num_words = max_words)

test_cases = [  
#     paragraph.get_text(),
    paragraph,
]

for case in test_cases:
    user_input = case
    user_sequences = tokenizer.texts_to_sequences([user_input])
    user_padded = tf.keras.preprocessing.sequence.pad_sequences(user_sequences, maxlen=max_sequence_length)

    from keras.models import load_model
    loaded_model = load_model('savedModel')  
    class_mapping={
    -1:'Negative',
    0:'Neutral',
    1:'Positive'}
    # Make predictions on user input 
    user_predictions = loaded_model.predict(user_padded)
    print(str(user_predictions))
    # Convert predictions to class labels (assuming it's a classification task)
    user_pred_classes = np.argmax(user_predictions, axis=1)
    # Print the predicted class
    print(f'Predicted Class: {class_mapping[user_pred_classes[0]]}')

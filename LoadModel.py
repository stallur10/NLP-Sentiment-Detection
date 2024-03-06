from keras.models import load_model
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import numpy as np

# loaded_model = tf.saved_model.load('savedModel')

# myData = "I hate bad stuff"
# predictions = loaded_model.predict(myData)
max_words = 10000  
max_sequence_length = 100  
tokenizer = Tokenizer(num_words=max_words)

test_cases = [
    
    "i hate black people",
    # "i love you",
    # "bro bro bro bro",
    # "cats are cool",
    # "i align myself with the nazi party i hate jews",
    # "fuck you"
    
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

# user_input = "i really really really hate hate hate hate hate black people"
# user_sequences = tokenizer.texts_to_sequences([user_input])
# user_padded = tf.keras.preprocessing.sequence.pad_sequences(user_sequences, maxlen=max_sequence_length)

# from keras.models import load_model
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
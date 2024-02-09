import pandas as pd
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Reddit_Data.csv")
df.rename({'clean_comment':'clean_text'}, axis=1, inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

X=pd.get_dummies(df)
y=df["category"]


# Preprocessing
import re
import string


def text_cleaning(text):
    """
    Clean the text using NLP and regular expressions
    
    text = Uncleaned text
    """
    text = re.sub(r'https?://\S+|www\.\S+', 'URL', text)
    text = re.sub(r'<.*?>', '', text)
    text = ''.join([char for char in text if char in string.printable])
    text = re.sub(r'@\S+', 'USER', text)
    table = str.maketrans('', '', string.punctuation)
    text = text.translate(table)
    text = ' '.join([word for word in text.split() if word not in stopwords.words("english")])
    return text

# df["clean_text"] = df["clean_text"].apply(text_cleaning)

print(df['clean_text'])

mapping = {0: 'Neutral', 1: 'Positive', -1: 'Negative'}
df['category'] = df['category'].map(mapping)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)











# # remove stopwords

# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords

# stop = set(stopwords.words("english"))

# def remove_stopwords(text):
#     filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
#     return " ".join(filtered_words)

# df['clean_comment'] = df['clean_comment'].fillna(df.mean()['clean_comment'])
# df["clean_comment"] = df.clean_comment.map(remove_stopwords)
# print(df.clean_comment)


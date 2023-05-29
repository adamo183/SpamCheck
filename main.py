import pandas as pd
import os
import pickle
from fastapi import FastAPI, Response, status
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

model_path = os.path.dirname(__file__) + '\\model.pickle'
vectorizer_path = os.path.dirname(__file__) + "\\vectorizer.pickle"

def create_model_spam(url):
    csv_data = pd.read_csv(url)
    random_data = csv_data.sample(frac=1)
    X_train, X_test, Y_train, Y_test = train_test_split(random_data['email'], random_data['label'],random_state=0)

    X_train_unicode = X_train.values.astype('U')
    X_test_unicode = X_test.values.astype('U')

    tf_vectorizer = CountVectorizer().fit(X_train_unicode)
    X_train_tf = tf_vectorizer.fit_transform(X_train_unicode)
    X_test_tf = tf_vectorizer.transform(X_test_unicode)
    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(X_train_tf, Y_train)
    return naive_bayes_classifier, tf_vectorizer

def read_model_from_file():
    vectorizer = pickle.load(open(vectorizer_path, 'rb'))
    model = pickle.load(open(model_path, 'rb'))
    return model,vectorizer

def save_model_to_file(classifier_to_save, vectorizer_to_save):
    pickle.dump(classifier_to_save, open(model_path, 'wb'))
    pickle.dump(vectorizer_to_save, open(vectorizer_path, "wb"))

#classifier, vectorizer = create_model_spam('data/spam.csv')
if os.path.isfile(model_path) and os.path.isfile(vectorizer_path):
    classifier, vectorizer = read_model_from_file()
else:
    classifier, vectorizer = create_model_spam('data/spam.csv')
    save_model_to_file(classifier, vectorizer)


app = FastAPI()

@app.post("/email/spam/check", status_code=200)
async def check_is_email_spam(data_to_check: str, response: Response):
    responseData = classifier.predict(vectorizer.transform([
        data_to_check
    ]))
    response.status_code = status.HTTP_200_OK
    return str(responseData[0])

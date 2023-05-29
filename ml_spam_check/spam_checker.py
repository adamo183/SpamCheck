import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


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

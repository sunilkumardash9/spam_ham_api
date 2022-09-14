import pandas as pd

#load dataset
df = pd.read_csv('/home/sunilkumardash9/Documents/spam_ham_api/spam.csv',encoding='ISO-8859-1')

#delete unwanted columns
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

#clean text
import regex as re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
def cleanText(text):
    text = re.sub(r"[\W_]",' ',text.lower()).replace('-','')
    words = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    new_text = " ".join(words)
    return new_text

#Building model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
#from xgboost import XGBRFClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

tfidf = TfidfVectorizer()


encode = LabelEncoder()
df['v1'] = encode.fit_transform(df['v1'])


X = df['v2'].apply(cleanText)
y = df['v1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
#switcher class for different estimators
class my_classifier(BaseEstimator,):
    def __init__(self, estimator=None):
        self.estimator = estimator
    def fit(self, X, y=None):
        self.estimator.fit(X,y)
        return self
    def predict(self, X, y=None):
        return self.estimator.predict(X,y)
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
    def score(self, X, y):
        return self.estimator.score(X, y)
#define estimators with parameters. (We have not taken other parameters to keep it simple)
from spamClassify import my_classifier
parameters = [
              {'clf':[LogisticRegression(max_iter=1000)],
              
               },
             {'clf':[RandomForestClassifier()],
            
             },
             {
               'clf':[DecisionTreeClassifier()],
             
             },
             
          ]

make_pipeline = Pipeline([('tfidf', tfidf), ('clf', my_classifier())])

grid = GridSearchCV(make_pipeline, parameters, cv=5)
grid.fit(X_train,y_train)

y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


#print(classification_report(y_test, y_pred))
print(f'Accuracy score is {accuracy*100}')


#import pickle
filename = 'spam_model'


from joblib import dump
dump(grid, filename)

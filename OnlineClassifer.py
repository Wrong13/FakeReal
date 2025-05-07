import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import pymorphy3
import nltk
from nltk.corpus import stopwords
import joblib
from datetime import datetime
import inspect
import sys
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec


class NewsClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.stopwords = None
        self.lemmatizer = pymorphy3.MorphAnalyzer()
    
    
    
    def first_train(self, df, save_path=None):
        src_path = Path(__file__).parent / "src"
        src_path.mkdir(exist_ok=True)
        nltk.download('stopwords', quiet=True)
        self.stopwords = set(stopwords.words('russian')).union(
            {'это', 'который', 'весь', 'такой', 'свой', 'наш', 'ваш', 'их'}
        )
        
        X = df['title']
        y = df['is_fake']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=list(self.stopwords)
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        
        self.model = PassiveAggressiveClassifier(
            random_state=42,
            early_stopping=True,
            max_iter=1000
        )
        self.model.fit(X_train_vec, y_train)
        
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        conf_mat = confusion_matrix(y_test, y_pred, labels=self.model.classes_)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.model.classes_,
                    yticklabels=self.model.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        save_path = src_path / "fake_news_model.mdl"

        if save_path:
            self.save_model(save_path)
        
        return accuracy
    
    def save_model(self, path):
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'stopwords': list(self.stopwords),
            'metadata': {
                'saved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'classes': self.model.classes_
            }
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"Модель сохранена в {path}")
    
    @classmethod
    def load_model(cls, model_path=None):
        if model_path is None:
            model_path = Path(__file__).parent / "src" / "fake_news_model.mdl"
        
        classifier = joblib.load(model_path)
        if not isinstance(classifier, cls):
            raise ValueError("Загруженный файл не является моделью NewsClassifier")
        return classifier

    def predict(self, text):
        
        text_vec = self.vectorizer.transform([text])  
        
        prediction = self.model.predict(text_vec)
        return 'Ложь' if prediction[0] else 'Так и есть'
    
    def partial_fit(self, X_new, y_new):
        X_vec = self.vectorizer.transform(X_new)
        self.model.partial_fit(X_vec, y_new, classes=self.model.classes_)
        print(f"Модель обновлена на {len(X_new)} примерах")
        self.save_model(path="src/fake_news_model.mdl")
        return self
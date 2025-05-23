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
#import pymorphy2
import nltk
from nltk.corpus import stopwords
import joblib

import OnlineClassifer

def main():
    s = 3
    if s == 0:
        classifer = OnlineClassifer.NewsClassifier.load_model()

        test_text = input()
        result = classifer.predict(test_text)
        print(f"Предсказание: '{test_text}' -> {result}")

    elif s == 1:
        src_path = Path(__file__).parent / "src"
        train_path = src_path / "train.tsv"
        df = pd.read_csv(train_path, sep='\t')

        
        classifier = OnlineClassifer.NewsClassifier() 
        accuracy = classifier.first_train(
            df=df,
            save_path="src/fake_news_model.mdl"  
        )

        print(f"Точность модели: {accuracy:.2f}")
    elif s == 2:
        src_path = Path(__file__).parent / "src"
        src_path.mkdir(exist_ok=True)
        df = pd.read_csv("src\\MyData.csv",encoding='cp1251')
        classifier = OnlineClassifer.NewsClassifier() 
        accuracy = classifier.first_train(
            df=df,
            save_path="src/fake_news_model.mdl"  
        )

        print(f"Точность модели: {accuracy:.2f}")
    elif s==3:
        df = pd.read_csv("src\\translated_parallel.csv")[:1200]
        df = df.drop(0, axis=0).reset_index(drop=True)
        df = pd.DataFrame(df)
        df['is_fake'] = 1
        df.columns = ['title','is_fake']
        df2 = pd.read_csv("src\\translated_parallel_True.csv")[:1200]
        df2 = df2.drop(0, axis=0).reset_index(drop=True)
        df2['is_fake'] = 0
        df2.columns = ['title','is_fake']
        df2 = pd.DataFrame(df2)
        
        rezultdf = pd.concat([df,df2],ignore_index=True)
        rezultdf = rezultdf.sample(frac=1).reset_index(drop=True)
        classifier = OnlineClassifer.NewsClassifier() 
        accuracy = classifier.first_train(
            df=rezultdf,
            save_path="src\\fake_news_model2.mdl"  
        )

        print(f"Точность модели: {accuracy:.6f}")

if __name__ == '__main__':
    main()
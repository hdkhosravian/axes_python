import app

import os
import re
import json
import joblib
import datetime
import operator
import pandas as pd
import numpy as np
from string import digits
from textblob import TextBlob

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from ..popular.keywords import top_feats_in_doc
from .constructor import set_global_vec, set_global_tn_df, set_global_tn_km, set_global_document_df

def train_articles(data):
    try:
        docs = data["articles"] 
        documents = text_make_clean(docs)
        # vec = TfidfVectorizer(stop_words=stopwords.words('english'), use_idf=True)
        vec = TfidfVectorizer(
            max_features=None, strip_accents='unicode',
            tokenizer=textblob_tokenizer, analyzer="word", token_pattern=r'\w{1,}',
            use_idf=1, min_df=0.01, smooth_idf=1, sublinear_tf=1,
            stop_words=stopwords.words('english')
        )

        matrix = vec.fit_transform(documents)
        df = pd.DataFrame(matrix.toarray(),
                          columns=vec.get_feature_names()).replace(np.nan, 0)

        number_of_clusters = 76 if len(documents) >= 150 else int(len(documents)/2) + 1
        km = KMeans(n_clusters=number_of_clusters)
        km.fit(matrix)

        results = pd.DataFrame.from_dict(docs, orient='columns')
        results['axes_world_category'] = km.labels_
        # results = pd.concat([results, df], axis=1)
        results = results.loc[:, ~results.columns.duplicated()]
        results["top_scores"] = df.apply(
            lambda row: top_feats_in_doc(
                matrix, vec.get_feature_names(), row, 100),
            axis=1
        )
        results['index'] = results.index
        
        df['index'] = results.index
        df['axes_world_category'] = km.labels_
        results = results.loc[:, ~results.columns.duplicated()]
        df = df.loc[:, ~df.columns.duplicated()]

        # path
        path = "trains/{}".format(data['report'])
        
        try:
            os.mkdir(path)
        except OSError as e:
            print("Creation of the directory %s failed" % path)
            print("error %s is" % e)
        else:
            print("Successfully created the directory %s " % path)

        # write dataframe
        df.to_parquet("{}/train_data_frame.parquet".format(path),
                      engine='pyarrow')
        results.to_parquet(
            "{}/documents.parquet".format(path), engine='pyarrow')

        # write sklearn model
        joblib.dump(km, "{}/kmeans_model.pk".format(path))

        # save documents
        with open("{}/documents.json".format(path), 'w') as f:
            json.dump(data, f)
        
        if data['report']:
            report_type = data['report']
        else:
            report_type = ""
        
        set_global_vec(report_type, vec)
        set_global_tn_df(report_type, df)
        set_global_tn_km(report_type, km)
        set_global_document_df(report_type, results)

        result = {
            'response': "training is done",
            'results': json.loads(
                results.drop(columns="body").to_json(orient='records')
            ),
        }
        return result, 200
    except Exception as e:
        print('error is: {}'.format(e))
        result = {
            'response': "an error happend"
        }
        return result, 400


def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words


def text_make_clean(texts):
    documents = []
    stemmer = WordNetLemmatizer()
    for sen in range(0, len(texts)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(texts[sen]))
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        # remove numbers
        document = re.sub(r'\d+', '', document)
        # Converting to Lowercase
        document = document.lower()
        document = document.replace('\n', '')
        # Lemmatization
        document = document.split()
        document = [stemmer.lemmatize(word, 'v') for word in document]

        document = ' '.join(document)
        documents.append(document)

    return documents

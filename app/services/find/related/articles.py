import re
import json
import numpy as np
import pandas as pd

from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from ...popular.keywords import top_feats_in_doc, top_mean_feats
from app.services.train.constructor import get_global_vec, get_global_tn_df, get_global_tn_km, get_global_document_df


def keyword_related_articles(keyword, data):
    keywords = data["keywords"]
    limit = data["limit"]
    report = data["report"]
    
    results = get_global_tn_df(report)

    try:
        results = results.reindex(columns=np.append(
            results.columns.values, [keyword]))
        results = results.loc[:, ~results.columns.duplicated()]
        results = results.reindex(columns=np.append(
            results.columns.values, keywords))
        results = results.loc[:, ~results.columns.duplicated()]
        results = results.replace(np.nan, 0)
        results = results.sort_values(by=keyword, ascending=False)
        results = results[results[keyword] > 0]
        results = results.loc[:, ~results.columns.duplicated()]
    except Exception as e:
        print('we have an error to find related articles to a keyword: {}', e)

    articles = get_global_document_df(report).loc[results.index.tolist()].head(limit)
    articles["keyword_score"] = 0

    # articles["keywords_score"] = results.apply(lambda row: row[keywords].mean(), axis=1)
    if len(results) > 0:
        articles["keyword_score"] = results[[keyword]]
    # json.loads(articles.to_json(orient='records'))

    result = {
        'articles': json.loads(
            articles[["id", "keyword_score"]].to_json(orient='records')
        ),
        'keywords_scores': json.loads(results[keywords].mean().to_json(orient='records')),
        'top_keywords': json.loads(top_mean_feats(get_global_tn_df(report), get_global_vec(report).get_feature_names(), top_n= 10))
    }
    return result, 200


def article_related_to_article(data):
    keywords = data["keywords"]
    limit = data["limit"]
    report = data["report"]
    body = data["body"]

    results = get_global_tn_df(report)
    try:
        results = results.reindex(columns=np.append(
            results.columns.values, keywords))
        results = results.loc[:, ~results.columns.duplicated()]
        results = results.replace(np.nan, 0)
        results = results.loc[:, ~results.columns.duplicated()]
    except Exception as e:
        print('we have an error to find related articles to a keyword: {}', e)

    document = text_make_clean(body)
    documents = get_global_document_df(report)
    vec = get_global_vec(report).transform([document])

    prediction = get_global_tn_km(report).predict(vec)

    articles = get_global_document_df(report)[documents["axes_world_category"] == int(
        prediction)].head(limit)

    result = {
        'articles_ids': json.loads(articles["id"].to_json(orient='records')),
        'keywords_scores': json.loads(results[keywords].mean().to_json(orient='records')),
        'top_keywords': json.loads(top_mean_feats(get_global_tn_df(report), get_global_vec(report).get_feature_names(), top_n= 10, is_text= True))
    }
    return result, 200

def article_related_articles(data):
    keywords = data["keywords"]
    limit = data["limit"]
    report = data["report"]
    category_id = data["category_id"]
    
    results = get_global_tn_df(report)
    try:
        results = results.reindex(columns=np.append(
            results.columns.values, keywords))
        results = results.loc[:, ~results.columns.duplicated()]
        results = results.replace(np.nan, 0)
        results = results.loc[:, ~results.columns.duplicated()]
    except Exception as e:
        print('we have an error to find related articles to a keyword: {}', e)

    documents = get_global_document_df(report)

    articles = get_global_document_df(report)[documents["axes_world_category"] == int(
        category_id)].head(limit)

    result = {
        'articles_ids': json.loads(articles["id"].to_json(orient='records')),
        'keywords_scores': json.loads(results[keywords].mean().to_json(orient='records')),
        'top_keywords': json.loads(top_mean_feats(get_global_tn_df(report), get_global_vec(report).get_feature_names(), top_n= 10, is_text= True))
    }
    return result, 200


def text_make_clean(text):
    documents = []
    stemmer = WordNetLemmatizer()
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(text))
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
    # Lemmatization
    document = document.split()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    return document

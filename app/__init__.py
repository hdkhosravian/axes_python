import glob
import joblib
import pandas as pd

from flask import Flask, json
from flask_bcrypt import Bcrypt

from flask_restplus import Api
from flask import Blueprint

from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from config.config import config_by_name
from .controllers.train_controller import api as train_articles_ns
from .controllers.find_related_controller import api as related_articles_ns
from .controllers.find_popular_controller import api as popular_keywords_ns
from .services.train.articles import train_articles

flask_bcrypt = Bcrypt()

blueprint = Blueprint('api', __name__)
api = Api(blueprint)

api.add_namespace(train_articles_ns, path='/train')
api.add_namespace(related_articles_ns, path='/related')
api.add_namespace(popular_keywords_ns, path='/popular')

documents_day_data_frame = None
documents_week_data_frame = None
documents_month_data_frame = None
documents_year_data_frame = None
documents_data_frame = None

train_articles_day_data_frame = None
train_articles_week_data_frame = None
train_articles_month_data_frame = None
train_articles_year_data_frame = None
train_articles_data_frame = None

train_articles_day_kmeans_model = None
train_articles_week_kmeans_model = None
train_articles_month_kmeans_model = None
train_articles_year_kmeans_model = None
train_articles_kmeans_model = None

vec_day_tfidf = None
vec_week_tfidf = None
vec_month_tfidf = None
vec_year_tfidf = None
vec_tfidf = None

def create_app(config_name):
    load_train_data()
    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])
    flask_bcrypt.init_app(app)

    return app

# add day path to routes and load all the global variables for different time reports
def load_train_data(report='all'):
    global train_articles_data_frame
    global train_articles_kmeans_model
    global documents_data_frame
    global vec_tfidf

    if train_articles_data_frame is None or train_articles_kmeans_model is None or vec_tfidf is None:
      path = "trains/{}".format(report)
      list_of_dir = glob.glob(path)
      try:
        latest_dir = max(list_of_dir)
        data = json.load(open(glob.glob("{}/documents.json".format(latest_dir))[0]))
        train_articles(data= data)
      except Exception as e:
        print("we have error in loading train files: {}".format(e))


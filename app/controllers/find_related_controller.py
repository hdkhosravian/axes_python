import json
import numpy as np

from flask import request
from flask_restplus import Resource

from ..utils.dto import RelatedDto
from ..services.find.related.articles import keyword_related_articles, article_related_articles, article_related_to_article

from ..services.train.constructor import get_global_tn_df

api = RelatedDto.api

@api.route('/articles/<keyword>')
class FindRelated(Resource):
    def post(self, keyword):
        """Find Related Articles base on keyword"""
        data = request.json
        return keyword_related_articles(keyword=keyword, data=data)

@api.route('/articles')
class FindRelatedArticles(Resource):
    def post(self):
        """Find Related Articles base on category"""
        data = request.json
        return article_related_articles(data=data)

@api.route('/article')
class FindRelatedToArticle(Resource):
    def post(self):
        """Find Related Articles base on custom article"""
        data = request.json
        return article_related_to_article(data=data)

@api.route('/article/<index>')
class ArticleKeywords(Resource):
    def post(self, index):
        """Find Popular Keywords"""
        data = request.json
        keywords = data["keywords"]

        results = get_global_tn_df('all')
        try:
            results = results.reindex(columns=np.append(
                results.columns.values, keywords))
            results = results.loc[:, ~results.columns.duplicated()]
            results = results.replace(np.nan, 0)
            results = results.loc[:, ~results.columns.duplicated()]
        except Exception as e:
            print('we have an error to find related articles to a keyword: {}', e)

        article = results.iloc[int(index)]

        result = {
            'keywords_scores': json.loads(article[keywords].to_json(orient='records'))
        }
        return result, 200  

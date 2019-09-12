import json
import numpy as np
from flask import request
from flask_restplus import Resource

from ..utils.dto import PopularDto
from ..services.popular.keywords import top_mean_feats
from ..services.train.constructor import get_global_vec, get_global_tn_df, get_global_tn_km, get_global_document_df

api = PopularDto.api

@api.route('/keywords')
class PopularKeywords(Resource):
    def post(self):
        """Find Popular Keywords"""
        data = request.json
        limit = data["limit"]
        report = data["report"]

        if 'category_id' in data:
            df = get_global_tn_df(report)[get_global_tn_df(report)["axes_world_category"] == int(data["category_id"])]
        else:
            df = get_global_tn_df(report) 

        result = {
            'top_keywords': json.loads(top_mean_feats(df, get_global_vec(report).get_feature_names(), top_n=limit)),
        }
        return result, 200

@api.route('/keywords/clients')
class ClientKeywords(Resource):
    def post(self):
        """Find Popular Keywords"""
        data = request.json
        keywords = data["keywords"]
        report = data["report"]

        results = get_global_tn_df(report)
        try:
            results = results.reindex(columns=np.append(
                results.columns.values, keywords))
            results = results.loc[:, ~results.columns.duplicated()]
            results = results.replace(np.nan, 0)
            results = results.loc[:, ~results.columns.duplicated()]
        except Exception as e:
            print('we have an error to find related articles to a keyword: {}', e)
    
        if 'category_id' in data:
            df = results[results["axes_world_category"] == int(data["category_id"])]
        else:
            df = results 

        result = {
            'keywords_scores': json.loads(df[keywords].mean().to_json(orient='records'))
        }
        return result, 200  

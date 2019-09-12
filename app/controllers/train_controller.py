from flask import request
from flask_restplus import Resource

from ..utils.dto import TrainDto
from ..services.train.articles import train_articles

api = TrainDto.api

@api.route('/')
class Trainarticles(Resource):
    def post(self):
        """Train base on articles"""
        data = request.json
        return train_articles(data=data)

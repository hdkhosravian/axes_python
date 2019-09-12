from flask_restplus import Namespace, fields

class TrainDto:
    api = Namespace('train_articles', description='train articles operations')

class RelatedDto:
    api = Namespace('related_keywords', description='find related articles base on keywords operations')

class PopularDto:
    api = Namespace('popular_keywords', description='find popular keywords')

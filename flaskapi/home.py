from flask import Flask, abort, request
from flask_restful import Resource, Api
from .utils import search_book

app = Flask(__name__)
api = Api(app)


class Home(Resource):
    def get(self):
        resp = "<h1>Welcome To VOTT Reviewer Service</h1> <br/> <p>Apis available:</p><ul><li>get: /</li><li>post: /{mlframework}/review</li></ul>"
        return resp


api.add_resource(Home, '/')


@app.errorhandler(404)
def not_found(e):
    return '', 404

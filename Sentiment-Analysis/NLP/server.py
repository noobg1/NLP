from sentiment_util import sentiment
from flask import Flask, request, Response
import json
import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


app = Flask(__name__)


@app.route('/')
def start():
	return "working"


@app.route('/q/<text>')
def query(text):
	print (text)
	result = (sentiment(tokenizer.tokenize(text)))
	print (result)	
	query_string = request.query_string
	
	data = {'result'  : str(result) }
	js = json.dumps(data)
	resp = Response(js, status=200, mimetype='application/json')
	return resp
	
if __name__ == '__main__':
    app.run(debug=True)
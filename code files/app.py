import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import re
import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 
from keras.preprocessing import sequence

app = Flask(__name__)
lstm_model = pickle.load(open('LSTM_model_2', 'rb'))
vocab = pickle.load(open('vocab', 'rb'))
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
	input1=request.form.get("statement")
	pp_data=preproc(input1)
	query=[]
	for i in pp_data.split():
		try:
			query.append((vocab.index(i))+1)
		except:
			continue
	#return render_template('index.html', sample_text='-> {}'.format(query))
	query=sequence.pad_sequences([query], maxlen=1400)
	prediction=lstm_model.predict_classes(query)
	prob=lstm_model.predict_proba(query)
	if(prediction==0):
		return render_template('index.html', prediction_text='The given statement is OPINION ', prediction_probability='with probability of {}'.format(prob[0][0]))
	else:
		return render_template('index.html', prediction_text='The given statement is FACT ',prediction_probability='with probability of {}'.format(prob[0][1]))

def preproc(data):
	stopword=""
	cleantext = " "
	stopword = set(stopwords.words('english'))
	for i in word_tokenize(data):
		if(any(j.isdigit() for j in i)): ## checking if the word has any numerics...
			continue
		i=i.lower()
		if (i not in stopword) and (len(i)>2):
			pattern1 = '[!.?$\[\]/\}#=<>"\*:,|_~;()^\']'
			pattern2 = '[\n\n]+'
			pattern3 = '[\ \ ]+'
			wout_sc = re.sub(pattern1,'',i) #removing special characters
			wout_el = re.sub(pattern2,'\n',wout_sc) # removing empty lines (which are greater than 2)
			wout_mspaces = re.sub(pattern3,' ',wout_el) # removing multiple spaces
			cleaned_text = wout_mspaces.strip()
			cleaned_text=lemmatizer.lemmatize(cleaned_text)
			if (i not in stopword) and (len(i)>2):
				cleantext = cleantext+cleaned_text+" "
	return cleantext.strip()

if __name__ == "__main__":
    app.run(debug=True)
# Serve model as a flask application

import pickle
import numpy as np
from flask import Flask, request
import json
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
from transformers import pipeline
import nltk
nltk.download('wordnet')
nltk.download('stopwords')

hvec= htc= mNB = None
app = Flask(__name__)

def load_transfomer_model():
    global hvec
    # model variable refers to the global variable
    with open('hvec.pkl', 'rb') as f:
        hvec = pickle.load(f)

def load_pred_model():
    global predictor
    global mNB
    # model variable refers to the global variable
    with open('mNB_model.pkl', 'rb') as f:
        mNB = pickle.load(f)
    with open('hug_model.pkl', 'rb') as f:
        hug = pickle.load(f)
    def hug_predict(x):
        try: return hug(x)[0]
        except: return np.nan
    predictor = hug_predict

global decide_func
decide_func = lambda df: 1 if (df.danger and df.Sentiment=="NEGATIVE") else 0

@app.route('/')
def home_endpoint():
    return 'Hello World!'

def predict_pipeline(df_data, content_col = "content"):
    def processing_text(series_to_process):
        new_list = []
        tokenizer = RegexpTokenizer(r'(\w+)')
        lemmatizer = WordNetLemmatizer()
        for i in range(len(series_to_process)):
            #TOKENISED ITEM(LONG STRING) IN A LIST
            dirty_string = (series_to_process)[i].lower()
            words_only = tokenizer.tokenize(dirty_string) #WORDS_ONLY IS A LIST THAT DOESN'T HAVE PUNCTUATION
            #LEMMATISE THE ITEMS IN WORDS_ONLY
            words_only_lem = [lemmatizer.lemmatize(i) for i in words_only]
            #REMOVING STOP WORDS FROM THE LEMMATIZED LIST
            words_without_stop = [i for i in words_only_lem if i not in stopwords.words("english")]
            #RETURN SEPERATED WORDS INTO LONG STRING
            long_string_clean = " ".join(word for word in words_without_stop)
            new_list.append(long_string_clean)
        return new_list
    cols = df_data.columns
    df_data["content_clean"] = processing_text(df_data[content_col])
    #GETTING PREDICTIONS
    X_test_tvec = hvec.transform(df_data["content_clean"]).todense()
    #ADDING PREDICTIONS AS A COLUMN IN THE DATAFRAME
    df_data["danger"] = pd.DataFrame(mNB.predict(X_test_tvec))
    df_data["Result"] = df_data[['content_clean']].applymap(predictor)
    df_data["Sentiment"] = df_data["Result"].map(lambda x: x['label'] if x != None else np.nan)
    df_data["Score"] = df_data["Result"].map(lambda x: x['score'] if x != None else np.nan)
    decide_func = lambda df: 1 if (df.danger and df.Sentiment=="NEGATIVE") else 0
    df_data["Risk"] = df_data.apply(decide_func, axis=1)
    return df_data[cols.append(pd.Index(['Risk']))]

@app.route('/predict_risk', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data = pd.DataFrame(request.get_json()['body'])  # Get data posted as a json
        df_predict = predict_pipeline(data)
    return df_predict.to_json()

if __name__ == '__main__':
    load_transfomer_model()  # load model at the beginning once only
    load_pred_model()
    app.run(host='0.0.0.0', port=80)
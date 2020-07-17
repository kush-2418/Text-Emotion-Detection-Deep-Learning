#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 22:52:40 2020

@author: kush
"""

'''
Keras == 2.3.1
tensorflow == 1.14.0
'''
from flask import Flask,render_template,request
import pickle
import os
from numpy import array

import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

from keras.preprocessing.sequence import pad_sequences
tokenizer = pickle.load(open('tokenizer.pkl','rb'))
sess = tf.Session()
graph = tf.get_default_graph()

set_session(sess)
model = load_model('emotion_cnn_model.h5')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


from flask import send_from_directory     

@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

def predict_emotion(text):
    maxlen = 75
    text = str(text)
    sequences = tokenizer.texts_to_sequences([text])
    pad_seq = pad_sequences(sequences, maxlen=maxlen)
    with graph.as_default():
        set_session(sess)
        pred = model.predict_classes(pad_seq)
    return pred


@app.route('/get_emotion',methods=["POST","GET"])

def get_emotion():
    if request.method=='POST':
        userText = request.form['message']
        predict = predict_emotion(userText)
        int2label = {
            0: 'Anger',
            1: 'Fear',
            2: 'Joy',
            3: 'Love',
            4: 'Sadness',
            5: 'Surprise'
            }
        emotion = int2label[int(predict)]

    return render_template('index.html', emotion = 'The Predicted Emotion is {}'.format(emotion))
    
if __name__ == '__main__':
    app.run()

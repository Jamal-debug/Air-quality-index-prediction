# -*- coding: utf-8 -*-
"""
Created on Tue May 19 13:33:41 2020

@author: Jamal
"""


from flask import Flask,render_template,request
import pickle
import numpy as np
model=pickle.load(open('model.pkl','rb'))
app=Flask(__name__,template_folder='template')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['post'])
def pre():
    data=[float(y) for y in request.form.values()]
    ndata=[np.array(data)]
    pdx=model.predict(ndata)
    pred=np.round(pdx[0],2)
    return render_template('home.html',pred='the Air Quality Index should be {} %'.format(pred),)
    #return jsonify(pred)

if(__name__ == '__main__'):
    app.run(debug=True)
    
    

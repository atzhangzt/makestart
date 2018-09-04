"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from FlaskDeepServer import app
from flask import request
import json
import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
from FlaskDeepServer import FImage

sess_dogcat = tf.Session()
fil =os.getcwd()
saver = tf.train.import_meta_graph('FlaskDeepServer/models/dogs-cats-model/dog-cat.ckpt-7975.meta')
saver.restore(sess_dogcat, 'FlaskDeepServer/models/dogs-cats-model/dog-cat.ckpt-7975')


@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )

@app.route('/dogcat', methods=['GET','POST'])
def upload_file():
    data =[]
    for file in request.files:
        ofile =  request.files.get(file)
        imgPath = os.path.join('C:\\Users\\MakeStart\\Pictures',ofile.filename)
        ofile.save(imgPath)
        
        images = FImage.getImages(imgPath,64)
        result = FImage.getResult(sess_dogcat,images,64,3)
        res_label = ['dog','cat']
        str = r'%s  %s  %s'%(file,result,res_label[result.argmax()])
        data.append(str)
    return json.dumps(data)
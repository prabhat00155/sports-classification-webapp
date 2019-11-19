import os
from statistics import mode, StatisticsError 
from PIL import Image
from flask import ( 
    Flask,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
import numpy as np
import json
import requests
from onnxruntime import InferenceSession
from torch.autograd import Variable
from torchvision import transforms

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return ('.' in filename and
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

@app.route('/')
def main():
    return render_template('home_url.html')

@app.route('/predict')
def predict():
    service_url = 'http://fb0d83b1-c0a6-441a-b6e2-09661b7b06b4.uksouth.azurecontainer.io/score'
    image_url = session.get('image_url', None) 
    test_sample = json.dumps({'data': [
       image_url 
    ]})
    test_sample = bytes(test_sample, encoding = 'utf8')
    headers = {'Content-Type':'application/json'}
    labels = []
    for i in range(3):
        resp = requests.post(service_url, test_sample, headers=headers)
        print("prediction:", resp.text)
        labels.append(resp.text.split(',')[0].split(':')[-1].strip(' \"'))
    try:
        label = mode(labels)
    except StatisticsError:
        label = labels[0]
    return render_template('result.html', file_name=image_url,
                           label=label)

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.form:
            flash('No file part')
            return redirect(request.url)
        uploaded_url = request.form['file']
        if uploaded_url == '':
            flash('No URL entered for uploading')
            return redirect(request.url)
        if uploaded_url:# and allowed_file(uploaded_url):
            session['image_url'] = uploaded_url
            flash('File uploaded successfully')
            return redirect('/predict') 
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return redirect(request.url)



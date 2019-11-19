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
from onnxruntime import InferenceSession
from torch.autograd import Variable
from torchvision import transforms
from werkzeug.utils import secure_filename

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
    return render_template('home.html')

@app.route('/predict')
def predict():
    class_names = [
        'RockClimbing',
        'badminton',
        'bocce',
        'croquet',
        'polo',
        'rowing',
        'sailing',
        'snowboarding'
    ]
    image_name =  os.path.join(app.config['UPLOAD_FOLDER'],
                               session.get('image_name', None))
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.47637546, 0.485785  , 0.4522678 ], [0.24692202, 0.24377407, 0.2667196 ])
    ])
    ort_session = InferenceSession('sports_classification-pretrained.onnx')
    raw_image = Image.open(image_name)
    labels = []
    for i in range(3):
        image = data_transforms(raw_image)
        image = image.numpy().reshape((1, *image.shape))
        res = ort_session.run(None, {'input.1': image})
        label_index = np.argmax(res[0])
        labels.append(class_names[label_index])
    try:
        label = mode(labels)
    except StatisticsError:
        label = labels[0]
    return render_template('result.html', file_name=image_name,
                           label=label)

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if uploaded_file and allowed_file(uploaded_file.filename):
            file_name = secure_filename(uploaded_file.filename) 
            uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
            session['image_name'] = file_name
            flash('File uploaded successfully')
            return redirect('/predict') 
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return redirect(request.url)



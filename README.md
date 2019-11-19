This project consists of two Python files corresponding to two WebApps, that
does the following:
1) sports\_classification.py: You may upload an sports image and see the sport
it represents.
2) sports\_classification\_url.py: You may paste an image URL that corresponds to
a sports image and see the sports being predicted.

In order to run these, we use Flask. First, let's install the requirements:
> pip install -r requirements.txt

Then, we set the following enviroment variables:
FLASK\_APP to the python file you would like to set as WebApp.
FLASK\_ENV to production or development.

To use sports\_classification.py, copy your trained onnx model to the same
directory as the python file and update the filename in the python code.

To use sports\_classification\_url.py, update the code with your deployed model's
URL.

Then, run the following:
> flask run

This will give you a URL which can be used to access the WebApp.

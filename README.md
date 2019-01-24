Rotten Fruit Detector
=====================

Introduction
------------

This project was part of my Coursera Course: Advanced Data Science Capstone. It will classify fruits (apples, bananas and oranges) and then will estimate whether they are fresh or rotten.
The dataset is not part of this project. It can be downloaded from https://www.kaggle.com/sriramr/fruits-fresh-and-rotten-for-classification

Dataset preparation
-------------------

The structure of the dataset folder should look like
```
dataset
   |---- apples
   |       |----- fresh
   |       |----- rotten
   |---- banana
   |       |----- fresh
   |       |----- rotten
   |---- oranges
   |       |----- fresh
   |       |----- rotten
   |---- combined (images from both fresh and rotten)
           |----- apples
           |----- oranges
           |----- bananas
```

For my dataset from kaggle, I have manually restructured the files and also cleared all rotated and transposed files as the data generator of Keras will already do that for me and even more advancedly.


Requirements
------------

I have developed this model with Python 3.6.x using Keras and Tensorflow. So be sure to install these dependencies using i.e. `pip`

Training
--------

Model definitions are in the training folder. The files can be opened in Visual Studio Code and treated as a Jupyter notebook. Nevertheless they can also be ran independently.
All command should be run from the root folder.

Let's train the first model: `fruitclassifier`
```
python training/fruitclassifier.py
```

A new folder will be created with the following structure:
```
results
    |----- fruits
              |---- model.yaml
              |---- weights.m5
```

Let's train the rottenness models:

```
FRUIT_CATEGORY="apples" python training/rottenness.py
FRUIT_CATEGORY="banana" python training/rottenness.py
FRUIT_CATEGORY="oranges" python training/rottenness.py
```

Now the models are trained.

Deployment
----------

Currently the model has to be deployed manually. Consider this the acceptance test.
In order to so copy the results directory into webapp/rottenfruitdetector/ as 'model'

Webapp
------
Run the webapp with
```
python manager.py runserver
```

In code of deployment elsewhere you may need to collect static files:
```
python manage.py collectstatic
```

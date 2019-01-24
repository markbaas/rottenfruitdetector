from pathlib import Path

import numpy as np
from keras import backend as K
from keras.models import model_from_yaml
from keras.preprocessing.image import img_to_array
from PIL import Image

here = Path(__file__).parent

img_width, img_height = 32, 32


def load_image(fd):
    im = Image.open(fd).convert('RGB')
    im = im.resize((img_width, img_height))
    x = img_to_array(im)
    x = np.expand_dims(x, axis=0)
    return x


def classify(x, model_name):
    model_path = here / "model" / model_name
    with open(model_path / "model.yaml") as f:
        model = model_from_yaml(f.read())
    model.load_weights(model_path / "weights.h5")

    print(model.predict_proba(x))
    print(model.predict_classes(x))

    category = model.predict(x)[0]

    K.clear_session()

    return category

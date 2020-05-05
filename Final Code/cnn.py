# -----------------------Imports-----------------------
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import cv2


# --------------------CNN LAYERS-----------------------
def cnnLayers():
    _input = Input((224, 224, 1))
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(_input)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(pool1)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(conv3)
    pool2 = MaxPooling2D((2, 2))(conv4)

    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(pool2)
    conv6 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(conv5)
    conv7 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(conv6)
    pool3 = MaxPooling2D((2, 2))(conv7)

    conv8 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(pool3)
    conv9 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv8)
    conv10 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv9)
    pool4 = MaxPooling2D((2, 2))(conv10)

    conv11 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(pool4)
    conv12 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv11)
    conv13 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv12)
    pool5 = MaxPooling2D((2, 2))(conv13)

    flat = Flatten()(pool5)
    dense1 = Dense(4096, activation="relu")(flat)
    dense2 = Dense(4096, activation="relu")(dense1)
    output = Dense(1000, activation="softmax")(dense2)

    vgg16_model = Model(inputs=_input, outputs=output)


cnnLayers()


# --------------------LOADING IMAGE--------------------
def _load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def _get_predictions(_model):
    f, ax = plt.subplots(1, 4)
    f.set_size_inches(80, 40)
    for i in range(4):
        ax[i].imshow(Image.open(imgs[i]).resize((200, 200), Image.ANTIALIAS))
    plt.show()

    f, axes = plt.subplots(1, 4)
    f.set_size_inches(80, 20)
    for i, img_path in enumerate(imgs):
        img = _load_image(img_path)
        preds = decode_predictions(_model.predict(img), top=3)[0]
        b = sns.barplot(y=[c[1] for c in preds], x=[c[2] for c in preds], color="gray", ax=axes[i])
        b.tick_params(labelsize=55)
        f.tight_layout()


# ---------------------RESNET50---------------------

from keras.applications.resnet50 import ResNet50

resnet50 = ResNet50(weights='imagenet', include_top=False)


def _get_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    resnet_features = resnet50.predict(img_data)
    return resnet_features


# --------------------TAINING DATA--------------------
def create_model():
    basepath = "D:/Emad/4th Elec/Second Term/Image Processing/Project/Dataset/Training set/"
    class1 = os.listdir(basepath + "Or/")
    class2 = os.listdir(basepath + "And/")

    data = {'Or': class1[::],
            'And': class2[::],
            'test': [class1[111], class2[111]]}

    features = {"Or": [], "And": [], "test": []}
    testimgs = []

    for label, val in data.items():
        for k, each in enumerate(val):
            if label == "test" and k == 0:
                img_path = basepath + "/Or/" + each
                testimgs.append(img_path)
            elif label == "test" and k == 1:
                img_path = basepath + "/And/" + each
                testimgs.append(img_path)
            else:
                img_path = basepath + label.title() + "/" + each
            feats = _get_features(img_path)
            features[label].append(feats.flatten())

    dataset = pd.DataFrame()
    for label, feats in features.items():
        temp_df = pd.DataFrame(feats)
        temp_df['label'] = label
        dataset = dataset.append(temp_df, ignore_index=True)

    # Features and Labels
    y = dataset[dataset.label != 'test'].label
    X = dataset[dataset.label != 'test'].drop('label', axis=1)

    # For saving data
    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

    model = MLPClassifier(hidden_layer_sizes=(100, 10))
    pipeline = Pipeline([('low_variance_filter', VarianceThreshold()), ('model', model)])
    pipeline.fit(X, y)

    print("Model Trained on pre-trained features")

    preds = pipeline.predict(features['test'])

    f, ax = plt.subplots(1, 2)
    for i in range(2):
        ax[i].imshow(Image.open(testimgs[i]).resize((200, 200), Image.ANTIALIAS))
        ax[i].text(10, 180, 'Predicted: %s' % preds[i], color='k', backgroundcolor='red', alpha=0.8)
    plt.show()
    pickle.dump(pipeline, open("pipeline.pickle", 'wb'))
    return pipeline  # return the model


pipeline = create_model()


# for testing a single image
def test_image(img_path):
    feats = _get_features(img_path)
    go = []
    go.append(feats.flatten())

    preds = pipeline.predict(go)

    plt.imshow(Image.open(img_path).resize((200, 200), Image.ANTIALIAS))
    plt.text(10, 180, 'Predicted: %s' % preds[0], color='k', backgroundcolor='red', alpha=0.8)
    plt.show()


test_image("D:/Emad/4th Elec/Second Term/Image Processing/Project/Dataset/Test set/And/A_35.JPG")


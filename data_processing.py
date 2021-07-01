# pip install mtcnn

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tensorflow.keras.models import model_from_json
from tensorflow.keras import preprocessing
from mtcnn.mtcnn import MTCNN

############# Show images ###############

# load images
def show_images(file_path):
  i = 1
  for filename in os.listdir(file_path):
    path = file_path + filename
    face = preprocessing.image.load_img(path)
    # plot
    plt.subplot(2, 7, i)
    plt.axis('off')
    plt.imshow(face)
    i += 1
  plt.show()

############## Extract faces from photoes ################

def extract_face(filename, target_size=(160,160)):
  # load image from file
  image = Image.open(filename)
  # convert to RGB, if needed
  image = image.convert("RGB")
  # convert to array
  img_arr = np.asarray(image)
  # create the detector, using default weights
  detector = MTCNN()
  # detect faces in the image
  results = detector.detect_faces(img_arr)
  # extract the bounding box from the first face
  x1, y1, width, height = results[0]["box"]
  # bug fix
  x1, y1 = abs(x1), abs(y1)
  x2, y2 = x1 + width, y1 + height
  # extract the face
  face = img_arr[y1:y2, x1:x2]
  # resize pixels to the model size
  image = Image.fromarray(face)
  image = image.resize(target_size)
  face_arr = np.asarray(image)
  return face_arr


###################### load images ##########################
def show_faces(file_path):
  i = 1
  for filename in os.listdir(file_path):
    path = file_path + filename
    face = extract_face(path)
    # plotting
    plt.subplot(2, 7, i)
    plt.axis('off')
    plt.imshow(face)
    i += 1
  plt.show()


################# Load all faces ###################
def load_faces(img_dir):
  faces = []
  for filename in os.listdir(img_dir):
    img_path = img_dir + filename
    face = extract_face(img_path)
    faces.append(face)
  return faces

################### load the weights of the pre-trained model ############################
def load_pre_trained_model(model_json, rf_model):
  with open(model_json, "r") as f:
    file_json = f.read()
    f.close()

  facenet_model = model_from_json(file_json)
  facenet_model.load_weights(rf_model)

  return facenet_model


################## images to encoding ###################
def encoding_images(img_dir, model):
  embedding_imgs = []
  faces = load_faces(img_dir)
  count = 0
  for i in faces:
    img = np.around(np.array(i) / 255.0, decimals=12)
    X_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(X_train)
    embedding = embedding / np.linalg.norm(embedding, ord=2)
    embedding_imgs.append(embedding)
    count += 1
  return embedding_imgs
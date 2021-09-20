import cv2
import dlib
import random
import numpy as np
from glob import glob
import matplotlib.pyplot as plt



def face_verification(img1, img2):
    detector = dlib.get_frontal_face_detector()
    shapePredictor = dlib.shape_predictor("/inputs/shape_predictor_5_face_landmarks.dat")
    model = dlib.face_recognition_model_v1("/inputs/dlib_face_recognition_resnet_model_v1.dat")

    # Ask the detector to find the bounding boxes of each face.
    detector_img1 = detector(img1, 1)
    detector_img2 = detector(img2, 1)

    img1_shape = shapePredictor(img1, detector_img1[0])
    img2_shape = shapePredictor(img2, detector_img2[0])

    # Get the aligned face image and show it
    img1_aligned = dlib.get_face_chip(img1, img1_shape)
    img2_aligned = dlib.get_face_chip(img2, img2_shape)

    img1_display = model.compute_face_descriptor(img1_aligned)
    img2_display = model.compute_face_descriptor(img2_aligned)

    # Convert to numpy array
    img1_display_arr = np.array(img1_display)
    img2_display_arr = np.array(img2_display)

    return img1_display_arr, img2_display_arr



def GetEuclideanDistance(original_face, test_face):
  euclidean_dist = original_face - test_face
  euclidean_dist = np.sum(np.multiply(euclidean_dist, euclidean_dist))
  euclidean_dist = np.sqrt(euclidean_dist)
  return euclidean_dist



def show_faces(path):
    multi_images = glob(path + "/**")
    r = random.sample(multi_images, 9)
    plt.figure(figsize=(20, 20))
    plt.subplot(331)
    plt.imshow(cv2.imread(r[0]))
    plt.axis('off')
    plt.subplot(332)
    plt.imshow(cv2.imread(r[1]))
    plt.axis('off')
    plt.subplot(333)
    plt.imshow(cv2.imread(r[2]))
    plt.axis('off')
    plt.subplot(334)
    plt.imshow(cv2.imread(r[3]))
    plt.axis('off')
    plt.subplot(335)
    plt.imshow(cv2.imread(r[4]))
    plt.axis('off')
    plt.subplot(336)
    plt.imshow(cv2.imread(r[5]))
    plt.axis('off')
    plt.subplot(337)
    plt.imshow(cv2.imread(r[6]))
    plt.axis('off')
    plt.subplot(338)
    plt.imshow(cv2.imread(r[7]))
    plt.axis('off')
    plt.subplot(339)
    plt.imshow(cv2.imread(r[8]))
    plt.axis('off')
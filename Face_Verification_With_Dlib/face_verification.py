import dlib
import os
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils

from tensorflow.keras import preprocessing
from utils_face import face_verification
from utils_face import GetEuclideanDistance
from utils_face import show_faces



if __name__ == "__main__":
    # Save all images in "IMAGES" folder
    ALL_IMAGE_PATH = os.path.join("/inputs", "IMAGES")

    image_dir = "/inputs/data/train"
    for directory in os.listdir(image_dir):
        for filename in os.listdir(os.path.join(image_dir, directory)):
            EX_PATH = os.path.join(image_dir, directory, filename)
            NEW_PATH = os.path.join(ALL_IMAGE_PATH, filename)
            os.replace(EX_PATH, NEW_PATH)

    # Plot images
    path = "/inputs/IMAGES"
    show_faces(path)

    PATH_IMG1 = ".../1a2a71a9fd194094b9f92cac91c6a7d6.jpg"
    PATH_IMG2 = ".../bigstock-Angelina-Jolie.jpg"
    # Load the image using dlib
    img1 = dlib.load_rgb_image(PATH_IMG1)
    img2 = dlib.load_rgb_image(PATH_IMG2)
    # Get image 1 and 2 from dlib detector
    img1_arr, img2_arr = face_verification(img1=img1, img2=img2)
    # Now get the euclidean distance to verify images
    euclidean_dist = GetEuclideanDistance(img1_arr, img2_arr)
    print(euclidean_dist)

    # We can set a threshold to specify the images are the same or not
    threshold = 0.6
    if euclidean_dist < threshold:
        print("They are the same person!")
    else:
        print("They are different person!")




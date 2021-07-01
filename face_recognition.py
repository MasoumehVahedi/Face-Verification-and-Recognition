import numpy as np
from data_processing import encoding_images

"""
                           ################# Face Recognition #####################

    In this section, we are going to  implement a face recognition system that takes as input an image,
    and figures out if it is one of the authorized persons (and if so, who).
    Unlike the face verification system, you will no longer get a person's name as one of the inputs.

"""


def face_recognition_fn(img_dir, database, model):
    """
       Arguments:
          img_dir -- path to an image
          database -- database containing image encodings along with the name of the person on the image
          model -- your Inception model instance in Keras

      Returns:
          min_dist -- the minimum distance between image_path encoding and the encodings from the database
          identity -- string, the name prediction for the person on image_path
    """
    encoding = encoding_images(img_dir, model)
    # Find the closest encoding, for instance 100
    min_dist = 100

    # Loop over the database dictionary's names and encodings
    for (name, encod) in database.items():
        # Compute L2 distance between the target "encoding" and the current "emb" from the database
        dist = np.linalg.norm(encod[0] - encoding[0])

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity
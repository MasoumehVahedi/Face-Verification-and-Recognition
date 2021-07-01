import numpy as np

from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
from data_processing import encoding_images



"""
                                   ############# Face Verification ############

    To face verification, we can use an encoding for each image in order to produce an accurate judgement 
    to verify two pictures of the same person.
    
    1- Encoding Face Images into a 128-Dimensional Vector

    In this project, we are going to use FaceNet model that was described by Florian Schroff, et al. 
    at Google in their 2015 paper titled “FaceNet: A Unified Embedding for Face Recognition and Clustering.”
    So, we use a ConvNet to calculate Encoding through a pre-trained facenet model. Some important things to do as follows:

       1- This network uses 160x160 dimensional RGB images as its input. 
          Specifically, a face image (or batch of  𝑚  face images) as a tensor of shape  (𝑚,𝑛𝐻,𝑛𝑊,𝑛𝐶)=(𝑚,160,160,3) 
       2- The input images are originally of shape 96x96, thus, you need to scale them to 160x160. 
          This is done in the img_to_encoding() function.
       3- The output is a matrix of shape  (𝑚,128)  that encodes each input face image into a 128-dimensional vector

"""


# Verify images : Next, we will verify each images using verify_fn() function.
def verify_fn(img_dir, identity, database, model):
    """
       Function that verifies if the person on the "image_path" image is "identity".

      Arguments:
          img_dir -- path to an image
          identity -- string, name of the person you'd like to verify the identity. Has to be an employee who works in the office.
          database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
          model -- your Inception model instance in Keras

      Returns:
          dist -- distance between the image_path and the image of "identity" in the database.
          door_open -- True, if the door should open. False otherwise.
    """
    encoding = encoding_images(img_dir, model)
    # Compute distance with identity's image
    # Note: there are 14 images in eache folders' name, so we should check the equal elemnts of encoding[i] and database[identity][i]
    dist = np.linalg.norm(encoding[0] - database[identity][0])
    # Open the door if dist < 0.7, else don't open
    if dist < 0.7:
        print("It's" + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

    return dist, door_open

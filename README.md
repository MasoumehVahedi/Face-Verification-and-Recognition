# Face Verification and Recognition with FaceNet

## Differentiate between face recognition and face verification
There are two categories for face recognition problems, including:

### 1- Face Verification:
Face verification is related to validate a claimed identity based on the image of a face, and either accepting or rejecting the identity claim (one-to-one matching problem).

### 2- Face Recognition: 
Face Recognition has to be compared with all the registered persons to recognize who is this person (one-to-many matching problem). 

## Content
In this case, in order to face verification, I have done two method:
1- "faceverification.py" : the pre-trained FaceNet model was used.
2- "faceverification.py" in FaceVerificationWithDlib folder: using dlib library and distance function, I have done a good result to verify images.
## Datasets
5 Celebrity Faces Dataset : https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset.

## Reference
1. FaceNet: A Unified Embedding for Face Recognition and Clustering
2. DeepFace: Closing the gap to human-level performance in face verification
3. https://github.com/davidsandberg/facenet


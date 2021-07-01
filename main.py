from data_processing import encoding_images
from data_processing import show_images
from data_processing import show_faces
from data_processing import load_pre_trained_model
from face_verification import verify_fn
from face_recognition import face_recognition_fn


if __name__ == "__main__":
    img_tarian_dir = "/content/data/val/ben_afflek/"
    img_val_dir = "/content/data/val/ben_afflek/"
    identity = "ben_afflek"

    model_json = "/content/drive/MyDrive/model.json"
    rf_model = "/content/drive/MyDrive/model.h5"

    # Plot images
    show_images(img_tarian_dir)
    # Extract faces
    show_faces(img_tarian_dir)

    # load the model
    facenet_model = load_pre_trained_model(model_json, rf_model)

    print(facenet_model.inputs)
    print(facenet_model.outputs)
    # model summary
    print(facenet_model.summary())


    # We are going to apply the FaceNet model for Face Verification
    # In this section, we want to create a database including one encoding vector for each person.
    database = {}
    database["ben_afflek"] = encoding_images("/content/data/train/ben_afflek/", facenet_model)
    database["elton_john"] = encoding_images("/content/data/train/elton_john/", facenet_model)
    database["jerry_seinfeld"] = encoding_images("/content/data/train/jerry_seinfeld/", facenet_model)
    database["madonna"] = encoding_images("/content/data/train/madonna/", facenet_model)
    database["mindy_kaling"] = encoding_images("/content/data/train/mindy_kaling/", facenet_model)

    # Face verification
    verify_fn(img_val_dir, identity, database, facenet_model)

    # Face Recognition
    face_recognition_fn(img_val_dir, database, facenet_model)


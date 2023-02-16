import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_path_list(root_path):
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''
    names = os.listdir(root_path)
    return names

def get_class_id(root_path, train_names):
    '''
        To get a list of train images and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        Returns
        -------
        list
            List containing all image in the train directories
        list
            List containing all image classes id
    '''
    image_list = []
    class_ids = []
    for id, person_name in enumerate(train_names):
        person_dir = root_path + "/" + person_name
        for person_img_sample in os.listdir(person_dir):
            img = cv2.imread(person_dir + "/" + person_img_sample)
            h, w, _ = img.shape
            ratio = 210 / w
            new_h = h * ratio
            img = cv2.resize(img, (210, int(new_h)))
            image_list.append(img)
            class_ids.append(id)
            # print(person_dir + "/" + person_img_sample, id)
    return image_list, class_ids
 

def detect_faces_and_filter(image_list, image_classes_list=None):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''
    cropped_face_images = []
    face_locations = []
    image_class_ids = []
    if image_classes_list is None:
        image_classes_list = [-1] * len(image_list)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    for img, id in zip(image_list, image_classes_list):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.2,
                                              minNeighbors=6)

        if len(faces) == 1:
            for (x, y, w, h) in faces:
                cropped_face = img[y:y + h, x:x + w]
                cropped_face_images.append(cropped_face)
                face_locations.append((x, y, w, h))
                image_class_ids.append(id)
        else:
            pass

    return cropped_face_images, face_locations, image_class_ids


def train(train_face_grays, image_classes_list):
    '''
        To create and train face recognizer object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_face_grays, np.array(image_classes_list))

    return recognizer

def get_test_images_data(test_root_path):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all loaded gray test images
    '''
    test_imgs = []

    for img_loc in os.listdir(test_root_path):
        img = cv2.imread(test_root_path + "/" + img_loc)
        h, w, _ = img.shape
        ratio = 510 / w
        new_h = h * ratio
        img = cv2.resize(img, (510, int(new_h)))
        test_imgs.append(img)

    return test_imgs
    
def predict(recognizer, test_faces_gray):
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''
    results = []
    for face in test_faces_gray:
        class_id, c = recognizer.predict(face)
        results.append(class_id)
        # print(class_id, c)

    return results

def get_wanted_status(prediction_result, train_names, wanted_names):
    '''
        To generate a list of wanted status (wanted or safe) from prediction results

        Parameters
        ----------
        prediction_result : list
            List containing all wanted results from given test faces
        test_image_list : list
            List containing all loaded test images
        wanted_names : list
            List containing all wanted names
        
        Returns
        -------
        list
            List containing all verification status from prediction results
    '''
    verification_statuses = []
    for i, pred in enumerate(prediction_result):
        name = train_names[pred]
        if name in wanted_names:
            verification_statuses.append([name, 0])
        else:
            verification_statuses.append([name, 1])

    return verification_statuses

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    '''
        To draw prediction results on the given test images and acceptance status

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            final result
    '''
    drawn_imgs = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    for ver, img, (x, y, w, h) in zip(predict_results, test_image_list, test_faces_rects):
        if ver[1] == 0:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)
            img = cv2.putText(img, ver[0], (x, y - 20), font, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            img = cv2.putText(img, "Wanted", (x, y + h + 50), font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            img = cv2.putText(img, ver[0], (x, y - 20), font, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            img = cv2.putText(img, "Safe", (x, y + h + 50), font, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
            
        img = cv2.resize(img, (250, 330))
        drawn_imgs.append([img, ver[1]])

    return drawn_imgs
    

def combine_and_show_result(image_list):
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    '''
    verified_imgs = []
    unverified_imgs = []
    white_img = np.zeros([250, 330, 3], dtype=np.uint8)
    white_img.fill(255)
    for img, ver in image_list:

        if ver == 1:
            verified_imgs.append(img)
        else:
            unverified_imgs.append(img)

    if len(verified_imgs) > len(unverified_imgs):
        while not len(verified_imgs) == len(unverified_imgs):
            unverified_imgs.append(white_img)
    elif len(verified_imgs) < len(unverified_imgs):
        while not len(verified_imgs) == len(unverified_imgs):
            verified_imgs.append(white_img)

    row2 = cv2.hconcat(verified_imgs)
    row1 = cv2.hconcat(unverified_imgs)

    result = cv2.vconcat([row1, row2])

    cv2.imshow("Results", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset/train"

    '''
        -------------------
        End of modifiable
        -------------------
    '''

    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset/test"
    wanted_names = ['Jackie Chan', 'Cho Yi-Hyun', 'Kim Se-jeong']

    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    verification_statuses = get_wanted_status(predict_results, train_names, wanted_names)
    predicted_test_image_list = draw_prediction_results(verification_statuses, test_image_list, test_faces_rects, train_names)
    
    combine_and_show_result(predicted_test_image_list)
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
    

def combine_and_show_result(image_list):
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
    '''

    

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
    train_root_path = ""

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
    test_root_path = ""
    wanted_names = []

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
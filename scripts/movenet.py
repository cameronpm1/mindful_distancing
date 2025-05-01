import numpy as np
import tensorflow as tf
from typing import Optional, Union

from scripts.util import *

class movenet():

    def __init__(
            self,
    ):
        
        self.interpreter = tf.lite.Interpreter(model_path="params/model.tflite")
        self.interpreter.allocate_tensors()

    def predict(
            self,
            image_input:Union[str,list[list[float]]]
    ):
        '''
        takes either loaded image or image file location, and returns pose angles
        computed from movenet prediction

        input
        -----
        image_input:Union[str,list[list[float]]]
            either the file location of the image, or loaded image matrix
        '''
        if type(image_input) is str:
            movenet_out = self.image_file_to_pose(image_input)
        else:
            movenet_out = self.image_to_pose(image_input)

        return self.transform_angles(movenet_out)

    def image_file_to_pose(
            self,
            image_dir:str
        ):
        '''
        take image file and computs movenet pose

        input
        -----
        image_dir:str
            file location of image, MUST BE JPEG

        output
        ------
        tuple[list[float]]
            predicted anchor positions in x,y, and confidence score
        '''
        # Load the input image.
        image = tf.io.read_file(image_dir)
        image = tf.image.decode_jpeg(image)

        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        input_image = tf.expand_dims(image, axis=0)
        input_image = tf.image.resize_with_pad(input_image, 192, 192)

        # Run model inference.
        x_keys, y_keys, scores = self.movenet_predict(input_image, scores=True)
        return x_keys, y_keys, scores #, np.append(x_keys,y_keys)

    def image_to_pose(
            self,
            image, 
        ):
        '''
        take image file and computs movenet pose

        input
        -----
        image_dir:list[list[float]]
            loaded image

        output
        ------
        tuple[list[float]]
            predicted anchor positions in x,y, and confidence score
        '''
        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        input_image = tf.expand_dims(image, axis=0)
        input_image = tf.image.resize_with_pad(input_image, 192, 192)

        # Run model inference.
        x_keys, y_keys, scores = self.movenet_predict(input_image, scores=False)
        return x_keys, y_keys, scores #, np.append(x_keys,y_keys)

    def movenet_predict(
            self,
            input_image, 
            scores=True
        ):
        """Runs detection on an input image.

        Args:
        input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.

        Returns:
        A [1, 1, 17, 3] float numpy array representing the predicted keypoint
        coordinates and scores.
        """
        # TF Lite format expects tensor type of uint8.
        input_image = tf.cast(input_image, dtype=tf.uint8)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        # Invoke inference.
        self.interpreter.invoke()
        # Get the model prediction.
        keypoints_with_scores = self.interpreter.get_tensor(output_details[0]['index'])
        
        if scores:
            x_keys = keypoints_with_scores[0, 0, :, 1]
            y_keys = keypoints_with_scores[0, 0, :, 0]
            scores = keypoints_with_scores[0, 0, :, 2]
            return x_keys, y_keys, scores
        else:
            x_keys = keypoints_with_scores[0, 0, :, 1]
            y_keys = keypoints_with_scores[0, 0, :, 0]
            return x_keys, y_keys

    def transform_angles(
            self,
            keys
        ):
        la = np.array([keys[0][15],keys[1][15]]) #left ankle
        ra = np.array([keys[0][16],keys[1][16]]) #right ankle
        lk = np.array([keys[0][13],keys[1][13]]) #left knee
        rk = np.array([keys[0][14],keys[1][14]]) #right knee
        lh = np.array([keys[0][11],keys[1][11]]) #left hip
        rh = np.array([keys[0][12],keys[1][12]]) #right hip
        ls = np.array([keys[0][5],keys[1][5]]) #left shoulder
        rs = np.array([keys[0][6],keys[1][6]]) #right shoulder
        le = np.array([keys[0][7],keys[1][7]]) #left elbow
        re = np.array([keys[0][8],keys[1][8]]) #right elbow

        left_side = ls - lh
        right_side = rs - rh
        left_arm = le - ls
        right_arm = re - rs
        left_uleg = lh - lk
        right_uleg = rh - rk
        left_lleg = la - lk
        right_lleg = ra - rk

        la_ang = angle_between_vectors(-left_side,left_arm) #left arm angle
        ra_ang = angle_between_vectors(-right_side,right_arm) #right arm angle
        lh_ang = angle_between_vectors(-left_uleg,left_side) #left hip angle
        rh_ang = angle_between_vectors(-right_uleg,right_side) #right hip angle
        lk_ang = angle_between_vectors(left_lleg, left_uleg) #left knee angle
        rk_ang = angle_between_vectors(right_lleg, right_uleg) #right knee angle

        return np.array([la_ang,ra_ang,lh_ang,rh_ang,lk_ang,rk_ang])
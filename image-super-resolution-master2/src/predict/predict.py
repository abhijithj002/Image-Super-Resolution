import os
import sys
import numpy as np
from time import time
from imageio import imwrite, imread
from logging import info, error
import cv2

#from utils.utils import browse_weights, load_model
sys.path.insert(0, '/home/deepquanty/projects/image-super-resolution-master/src/utils/')
#import utils
from utils import get_parser, load_model, load_configuration, browse_weights
#sys.path.insert(0, '/home/deepquanty/projects/image-super-resolution-master/src/predict/')



class Predictor:
    """The predictor class handles prediction, given an input model.
    Reads input files from the folder specified in config.json.
    Saves results in output folder specified in config.json.

    Can receive a path for the weights or can let the user browse through the
    weights directory for the desired weights.
    """
    def __init__(self, test_arguments):
        # Select pre-trained weights
        #self.weights_path, _ = browse_weights(weights_path=test_arguments['weights_path'])
        self.weights_path = '/home/ubuntu/Documents/image-super-resolution-master2/weights/sample_weights/DIV2K_E086_X2_D20G64C6.hdf5'
        self.results_folder = test_arguments['results_folder']
        self.test_folder = test_arguments['test_folder']
        self.verbose = test_arguments['verbose']
        self.extensions = ('.jpeg', '.jpg', '.png')  # file extensions that are admitted
        # Create results folder
        print("#################################################################")
        print(sys.path)
        if not os.path.exists(self.results_folder):
            info('Creating directory for results:\n', self.results_folder)
            os.mkdir(self.results_folder)

    def load_weights(self):
        if self.weights_path is not None:
            self.model.rdn.load_weights(self.weights_path)
            info('>>> Loaded weights from')
            info(self.weights_path)
        else:
            error(' invalid weights path')
            raise

    def get_predictions(self, model):
        self.model = model
        self.load_weights()
        file_ls = os.listdir(self.test_folder)
        img_ls = [file for file in file_ls if file.endswith(self.extensions)]
        # Predict and store
        print("Printing IMG_LS")
        print(img_ls)
        for img_name in img_ls:
            input_path = os.path.join(self.test_folder, img_name)
            output_path = os.path.join(self.results_folder, img_name)
            info('\n>>> Processing', input_path)
            start = time()
            sr_img = self.forward_pass(input_path)[0]
            end = time()
            info('Elapsed time', end - start)
            info('Result in', output_path)
            imwrite(output_path, sr_img)

    def forward_pass(self, file_path):
        print("File Path")
        print(file_path)
        #lr_img = imread(file_path)
        lr_img = cv2.imread(file_path)
        print(lr_img.shape)
        if lr_img.shape[2] == 3:
            lr_img = lr_img / 255.0
            lr_img = np.expand_dims(lr_img, axis=0)
            sr_img = self.model.rdn.predict(lr_img)
            return np.clip(sr_img, 0, 1) * 255
        else:
            error(' {} is not an image with 3-dim.'.format(file_path))



    

import os
import time

from config import *
from data_processing import *

class SegmentationModel:
    def __init__(self):
        self.image_width = IMAGE_WIDTH
        self.image_height = IMAGE_HEIGHT
        current_dir = os.getcwd()
        self.model_path = current_dir + MODEL_PATH

        # dictionary with:
        # "cateogory_name": [index, (R, G, B)]
        # index is pixel intensity for segmentation categories (e.g. for deeplab format)
        self.mask_dict = MASK_DICT
        self.class_items = self.mask_dict.items()
        self.num_class = len(self.class_items)

        self.model = deep_model

        if (self.model_path):
            self.model.load_weights(self.model_path)


    def load_model(self, model_path = None):
        if model_path is not None:
            self.model_path = model_path
        if (self.model_path):
            self.model.load_weights(self.model_path)


    def run_model_on_image(self, input_image_name, result_path):
        os.makedirs(result_path, exist_ok=True)
        self.input_image = input_image_name

        input_fname = os.path.basename(input_image_name)
        img, raw, scale_coef = read_and_preprocess_image(test_path=None,
                                                         filename=self.input_image,
                                                         target_size=(self.image_height, self.image_width))
        start_time = time.time()
        results = self.model.predict(img, verbose=0)
        end_time = time.time()
        duration = end_time - start_time

        out_file_name = os.path.join(result_path, input_fname.split('.')[0] + '.png')
        img = label_visualize_precise(mask_dict=self.mask_dict,
                                      img=results[0],
                                      scale_coef=scale_coef)
        io.imsave(out_file_name, img)
        print("Processed ( %.3f s) :" % (duration) + out_file_name)

    def run_model_on_folder(self, input_folder, result_path):
        os.makedirs(result_path, exist_ok=True)

        file_list = os.listdir(input_folder)

        for count, filename in enumerate(sorted(file_list)):
            self.input_image = filename

            img, raw, scale_coef = read_and_preprocess_image(test_path=input_folder,
                                                             filename=self.input_image,
                                                             target_size=(self.image_height, self.image_width))
            start_time = time.time()
            results = self.model.predict(img, verbose=0)
            end_time = time.time()
            duration = end_time - start_time

            out_file_name = os.path.join(result_path, filename.split('.')[0] + '.png')
            img = label_visualize_precise(mask_dict=self.mask_dict,
                                          img=results[0],
                                          scale_coef=scale_coef)
            io.imsave(out_file_name, img)
            print("Processed ( %.3f s) :" % (duration) + out_file_name)


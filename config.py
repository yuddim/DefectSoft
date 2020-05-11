
##----------- GPU selection ---------------------------------------------

GPU_ID = "0"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

from model import *


#GPU ID(s) used for training, evaluation or testing. If you want use multiple GPUs print "0,1,2"

##----------- Script mode selection ---------------------------------------------

TRAIN_FLAG = False  # True if we want train new model or tune pretrained model from MODEL_PATH
                    # on data from TRAIN_PATH and VAL_PATH
TUNE_FLAG = False   # True if we want to tune pretrained model from MODEL_PATH on data from TRAIN_PATH and VAL_PATH
EVALUATION_FLAG = False # True if we want to evaluate pretrained model on data from VAL_PATH
#If TRAIN_FLAG == False and TUNE_FLAG == False and EVALUATION_FLAG == False
# prediction results of pretrained model from MODEL_PATH on TEST_PATH will be calculated and saved to RESULT_PATH

##----------- Class pallete -----------------------------------------------------

# dictionary with:
# "cateogory_name": [index, (R, G, B)]
# index is pixel intensity for segmentation categories (e.g. for deeplab format)
# MASK_DICT = {
#     "background": [0, (0, 0, 0)],
#     "car": [1, (255, 255, 255)]
# }

# MASK_DICT = {
#     "background": [0, (0, 0, 0)],
#     "1.Vpadini": [1, (255, 255, 0)], #00ffff
#     "2.Vzdutie": [2, (153, 255, 204)],#"ccff99",  # -
#     "3.Skladki": [3, (6, 89, 182)],#"b65906",  # -
#     "4.Zaplatki": [4, (0, 255, 255)],#"ffff00",  # not in audi #-
#     #"5.Otslaivanie_kovra": [5, (0, 0, 255)],#"ff0000",
#     "6.Razrivi_kovra": [5, (255, 193, 255)]#,#"ffc1ff",  # not in audi
#     #"7.Otsutstvie_kovra": [7, (255, 0, 128)],#"8000ff",
#     #"8.Gribok/Mox": [8, (30, 30, 90)],#"5a1e1e",  # not in audi #-
#     #"9.Rastreskivanie": [9, (255, 153, 204)],#"cc99ff",
#     #"10.Spolzanie_kovra": [10, (37, 193, 255)]#"ffc125",
# }

# MASK_DICT = {
#     "background": [5, (255, 255, 255)],
#     "1.Vpadini": [0, (255, 255, 0)], #00ffff
#     "2.Vzdutie": [1, (153, 255, 204)],#"ccff99",  # -
#     "3.Skladki": [2, (6, 89, 182)],#"b65906",  # -
#     "4.Zaplatki": [3, (0, 255, 255)],#"ffff00",  # not in audi #-
#     #"5.Otslaivanie_kovra": [5, (0, 0, 255)],#"ff0000",
#     "6.Razrivi_kovra": [4, (255, 193, 255)]#,#"ffc1ff",  # not in audi
#     #"7.Otsutstvie_kovra": [7, (255, 0, 128)],#"8000ff",
#     #"8.Gribok/Mox": [8, (30, 30, 90)],#"5a1e1e",  # not in audi #-
#     #"9.Rastreskivanie": [9, (255, 153, 204)],#"cc99ff",
#     #"10.Spolzanie_kovra": [10, (37, 193, 255)]#"ffc125",
# }
MASK_DICT = {
    "background": [5, (255, 255, 255)],
    "1.Vpadini": [0, (0, 0, 128)],
    "2.Vzdutie": [1, (0, 128, 0)],
    "3.Skladki": [2, (255, 165, 0)],
    "4.Zaplatki": [3, (255, 192, 203)],
    "6.Razrivi_kovra": [4, (179, 157, 219)]
}


##----------- Paths settings  --------------------------------------------------

# Dataset tree is
# TRAIN_PATH -- IMAGES_FOLDER_NAME
#            |- MASKS_FOLDER_NAME
#
# VAL_PATH -- IMAGES_FOLDER_NAME
#          |- MASKS_FOLDER_NAME

# TRAIN_PATH = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/car_segm_dataset/train'
# VAL_PATH = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/car_segm_dataset/test'
#
# #RESULT_PATH = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/car_segm_dataset/results'
# RESULT_PATH = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/car_segm_dataset/raw_web_results'
#
# #TEST_PATH = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/car_segm_dataset/test/images'
# TEST_PATH = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/car_segm_dataset/raw_web'
#
#
# #MODEL_PATH = '/models/unet_light_mct2020-01-20-18-32-22.31-tloss-0.5139-tdice-0.7674-vdice-0.7584_car_segm.hdf5'
# MODEL_PATH = '/models/unet_light_mct2020-01-21-12-31-11.14-tloss-0.5585-tdice-0.7510-vdice-0.7397_car_segm.hdf5'
#
# TRAIN_PATH = '/datasets/Roof_defects_dataset/dataset/augmented_dataset/train'
# VAL_PATH = '/datasets/Roof_defects_dataset/dataset/augmented_dataset/test'
# RESULT_PATH = "/datasets/Roof_defects_dataset/dataset/results_multiclass_unet_test_1"
# TEST_PATH = "/datasets/Roof_defects_dataset/dataset/augmented_dataset/test/source"

TRAIN_PATH = '/home/cds-y/Datasets/roof_defects_dataset/augmented_dataset/train'
VAL_PATH = '/home/cds-y/Datasets/roof_defects_dataset/augmented_dataset/test'
RESULT_PATH = "/home/cds-y/Datasets/roof_defects_dataset/unetmct-cce-augm_nearest-unet_light_mct-2020-01-25-22-49-04.100-tloss-0.1853-vloss0.2672"
TEST_PATH = "/home/cds-y/Datasets/roof_defects_dataset/augmented_dataset/test/source"

SAVE_MODEL_DIR = 'models'
MODEL_PATH = '/models/unet-cce+sdice-2020-02-03-12-20-13.112-tloss-0.2804-tdice-0.7755-vdice-0.5217.hdf5'

TENSOR_BOARD_LOGS_PATH = './logs_tb/cce+sdice+unet+augm'

# IMAGES_FOLDER_NAME = 'images'
# MASKS_FOLDER_NAME = 'masks'

IMAGES_FOLDER_NAME = 'source'
MASKS_FOLDER_NAME = 'mask'

##----------- Augmentor settings  ---------------------------------------------

AUGMENT_CONFIG = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='constant',
                    cval = 0)
#AUGMENT_CONFIG = dict()

##----------- Calculate number of classes on base of MASK_DICT  ---------------

n_classes = len(MASK_DICT.items())

##----------- Model input image size-------------------------------------------

# IMAGE_WIDTH = 128
# IMAGE_HEIGHT = 128
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 576

##-------------- Set model  ---------------------------------------------------

deep_model = unet(input_size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                  n_classes=n_classes)

##-------------- Compile model or no -------------------------------------------

NO_COMPILE = False

##-------------- Set model loss function ---------------------------------------

#loss = categorical_crossentropy_loss

loss = loss_function_multilabel(num_labels=n_classes, smooth=1e-5, sparse_thresh=10, coefs=[1, 1.44, 1.49, 4.4, 44.9, 62.2])

##-------------- Set model metrics ---------------------------------------------

metrics = [sparsed_dice_loss_multilabel(num_labels=n_classes,
                                        smooth=1e-5,
                                        sparse_thresh=10,
                                        coefs=[1, 1.44, 1.49, 4.4, 44.9, 62.2])]
#metrics = [dice_0]
for channel_id in range(n_classes):
    # metrics.append(iou_metric_softmax(smooth=1e-5, channel_id=channel_id))
    metrics.append(dice_sparsed_metric(smooth=1e-5, thresh=10, channel_id=channel_id))

##-------------- Set model optimizer --------------------------------------------

LEARNING_RATE = 1e-4  # Starting learning rate for default optimizer - Adam
optimizer = Adam(lr=LEARNING_RATE)

##-------------- Set learning rate scheduler-------------------------------------

# after the first EPOCHS_SCHEDULE_STEP formula for Learning rate update is used:
# NEW_LEARNING_RATE = LEARNING_RATE *
#                     tf.math.pow(DROP_SCHEDULE_COEF, tf.math.floor((1 + epoch) / EPOCHS_SCHEDULE_STEP))
DROP_SCHEDULE_COEF = 1.0
EPOCHS_SCHEDULE_STEP = 10.0

##----------- Training parameters  ----------------------------------------------

N_EPOCHS = 100
BATCH_SIZE = 1      #Batch size for training
EVAL_BATCH_SIZE = 1  #Batch size for validation, evaluation and testing


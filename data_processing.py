from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import cv2
import time
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from shutil import copyfile

from config import *

mask_dict = MASK_DICT

def convert_num_mask_to_color(dir_path_source, dir_path_target, mask_dict, is_gray = True):
    os.makedirs(dir_path_target, exist_ok = True)
    for count, filename in enumerate(sorted(os.listdir(dir_path_source))):
        img = io.imread(os.path.join(dir_path_source,filename),as_gray = is_gray)
        class_items = mask_dict.items()
        new_mask = np.zeros((img.shape[0], img.shape[1], 3))
        for key, value in class_items:
            class_name = key
            class_index = value[0]
            class_color = value[1]
            new_mask[img == class_index] = class_color

        cv2.imwrite(dir_path_target + '/' + filename, new_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(str(count) + ':' + os.path.join(dir_path_source,filename))

def dataset_preparation(dir_path_source_masks, dir_path_source_images, dir_path_target, mask_dict,
                        size_threshold = 20, ratio = 10, is_gray = True):
    os.makedirs(dir_path_target, exist_ok = True)
    os.makedirs(dir_path_target+'/train', exist_ok=True)
    os.makedirs(dir_path_target+'/test', exist_ok=True)
    os.makedirs(dir_path_target + '/too_small', exist_ok=True)
    os.makedirs(dir_path_target + '/train'+'/images', exist_ok=True)
    os.makedirs(dir_path_target + '/train' + '/masks', exist_ok=True)
    os.makedirs(dir_path_target + '/test'+'/images', exist_ok=True)
    os.makedirs(dir_path_target + '/test' + '/masks', exist_ok=True)
    os.makedirs(dir_path_target + '/too_small'+'/images', exist_ok=True)
    os.makedirs(dir_path_target + '/too_small'+ '/masks', exist_ok=True)

    full_sorted_list = sorted(os.listdir(dir_path_source_masks))
    full_count = len(full_sorted_list)
    no_file_count = 0
    bad_crop_count = 0

    for count, filename in enumerate(full_sorted_list):
        img = io.imread(os.path.join(dir_path_source_masks,filename),as_gray = is_gray)
        class_items = mask_dict.items()
        height = img.shape[0]
        width = img.shape[1]
        new_mask = np.zeros((height, width, 3))
        for key, value in class_items:
            class_name = key
            class_index = value[0]
            class_color = value[1]
            new_mask[img == class_index] = class_color

        target_path = dir_path_target
        if(width < size_threshold or height < size_threshold):
            target_path = target_path + '/too_small'
        else:
            #if(count < train_part*full_count):
            if (count % ratio != 0):
                target_path = target_path + '/train'
            else:
                target_path = target_path + '/test'

        image_file_name = filename.split('.')[0] + '.jpg'
        source_image_file_name = dir_path_source_images + '/' + image_file_name
        error_message = ''

        if (os.path.isfile(source_image_file_name)):
            (width_image, height_image) = Image.open(source_image_file_name).size
            if(width == width_image and height==height_image):
                cv2.imwrite(target_path + '/masks/' + filename, new_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                copyfile(source_image_file_name, target_path + '/images/' + image_file_name)
            else:
                bad_crop_count += 1
                error_message = 'bad_crop'
        else:
            no_file_count += 1
            error_message = 'no_file'

        print(error_message + str(count) + ':' + os.path.join(dir_path_source_masks,filename))
    print('no_file_count: ', no_file_count)
    print('bad_crop_count: ', bad_crop_count)

def integral_histogram(histogram):
    new_histogram = np.zeros((len(histogram), 1))
    for value in range(len(histogram)):
        if value == 0:
            continue
        new_histogram[value] = new_histogram[value-1]+histogram[value]
    return new_histogram


def get_mask_statistics(dir_path_source):
    width_histogram = np.zeros((4000, 1))
    height_histogram = np.zeros((4000, 1))
    max_size_histogram = np.zeros((4000, 1))
    x = np.arange(4000)
    for count, filename in enumerate(sorted(os.listdir(dir_path_source))):
        mask_size = Image.open(os.path.join(dir_path_source,filename)).size
        (width, height) = mask_size
        width_histogram[width] += 1
        height_histogram[height] += 1
        max_size_histogram[max(width,height)] += 1
    max_size = 500
    fig = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    ax_1 = fig.add_subplot(211)
    ax_1.plot(x[:max_size], integral_histogram(width_histogram[:max_size]), color='red', linewidth=2)
    ax_1.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax_1.set(title='width_histogram')
    ax_1.grid(which='major',
            color='k')

    ax_2 = fig.add_subplot(212)
    ax_2.plot(x[:max_size], (max_size_histogram[:max_size]), color='red', linewidth=2)
    ax_2.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax_2.set(title='max_size_histogram')
    ax_2.grid(which='major',
              color='k')
    plt.show()

# get_mask_statistics(dir_path_source = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/car_segm_dataset/train/images')

# convert_num_mask_to_color(
#     dir_path_source = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/masks',
#     dir_path_target = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/color_masks',
#     mask_dict = mask_dict,
#     is_gray = True)

# dataset_preparation(
#     dir_path_source_masks = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/masks',
#     dir_path_source_images = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/images',
#     dir_path_target = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/car_segm_dataset',
#     mask_dict = mask_dict,
#     size_threshold = 20,
#     ratio = 10,
#     is_gray = True)

# dataset_preparation(
#     dir_path_source_masks = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/crop_train_masks',
#     dir_path_source_images = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/crop_train_images',
#     dir_path_target = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/car_segm_coco_dataset',
#     mask_dict = mask_dict,
#     size_threshold = 20,
#     ratio = 10,
#     is_gray = True)



def train_generator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "color",
                    mask_color_mode = "color",image_save_prefix  = "images",mask_save_prefix  = "masks",
                    flag_multi_class = False, num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1,
                    mask_dict = None):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img, mask, mask_dict)
        yield (img,mask)

def adjustData(img, mask, mask_dict=None):
    img = img / 255
    class_items = mask_dict.items()
    num_class = len(class_items)
    batch_size = mask.shape[0]
    new_mask = np.zeros((batch_size, mask.shape[1], mask.shape[2], num_class))

    #в эту функцию приходит не одна маска, а батч масок
    for key, value in class_items:
        class_name = key
        class_index = value[0]

        label_color = value[1]

        min_color = (label_color[0] - 5, label_color[1] - 5, label_color[2] - 5)
        max_color = (label_color[0] + 5, label_color[1] + 5, label_color[2] + 5)
        for batch_id in range(batch_size):
            new_mask[batch_id, :, :, class_index] = cv2.inRange(mask[batch_id], min_color, max_color)

    mask = new_mask
    mask = mask / 255
    # elif(np.max(img) > 1):
    #     img = img / 255
    #     mask = mask /255
    #     mask[mask > 0.5] = 1
    #     mask[mask <= 0.5] = 0
    return (img,mask)

def test_generator(test_path, target_size = (256,256), as_gray = False):
    for filename in sorted(os.listdir(test_path)):
        start_time = time.time()
        img = io.imread(os.path.join(test_path,filename),as_gray = as_gray)
        img = img / 255
        img = cv2.resize(img,(target_size[1],target_size[0]))
        img = np.reshape(img,(1,)+img.shape)
        end_time = time.time()
        duration = end_time - start_time
        with open('logs/timeread.txt', "a") as log_file:
            log_file.write("Imread, s: " + str(duration) + "\n")
        yield img

def read_and_preprocess_image(test_path,filename,target_size = (256,256)):
    if test_path is None:
        raw = io.imread(filename)
    else:
        raw = io.imread(os.path.join(test_path, filename))
        #raw_ = cv2.imread(filename)
    #raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    img = raw / 255
    img = cv2.resize(img, (target_size[1], target_size[0]))
    img = np.reshape(img, (1,) + img.shape)
    scale_coef = (raw.shape[0]/target_size[0], raw.shape[1]/target_size[1])

    return img, raw, scale_coef

def label_visualize_smoothed(mask_dict, img, scale_coef = (1,1)):
    class_items = mask_dict.items()
    num_class = len(class_items)
    (out_h, out_w) = (int(img.shape[0] * scale_coef[0]), int(img.shape[1] * scale_coef[1]))
    img_out = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for key, value in class_items:
        class_name = key
        class_index = value[0]
        label_color = value[1]
        threshold = value[2]

        mask = img[:,:,class_index]

        img_out[mask > threshold] = np.array(label_color)

    img_out_resized = cv2.resize(img_out, (out_w, out_h))

    return img_out_resized

def label_visualize_precise(mask_dict, img, scale_coef = (1,1)):
    class_items = mask_dict.items()
    num_class = len(class_items)
    (out_h, out_w) = (int(img.shape[0] * scale_coef[0]), int(img.shape[1] * scale_coef[1]))
    img_out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    mask = np.argmax(img, axis=-1)
    for key, value in class_items:
        class_name = key
        class_index = value[0]
        label_color = value[1]
        #threshold = value[2]

        #mask = img[:,:,class_index]
        #mask[mask > threshold] = 255
        #mask[mask <= threshold] = 0
        channel_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        channel_mask[mask == class_index] = 255
        mask_resized = cv2.resize(channel_mask, (out_w, out_h))

        img_out[mask_resized > 128] = np.array(label_color)

    return img_out

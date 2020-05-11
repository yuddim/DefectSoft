import os
import numpy as np
from PIL import Image
import cv2

mask_dict_1 = {
    "background": [0, (0, 0, 0)],
    "1.Vpadini": [1, (255, 255, 0)],
    "2.Vzdutie": [2, (153, 255, 204)],
    "3.Skladki": [3, (6, 89, 182)],
    "4.Zaplatki": [4, (0, 255, 255)],
    "5.Otslaivanie_kovra": [5, (0, 0, 255)],
    "6.Razrivi_kovra": [5, (255, 193, 255)],
    "7.Otsutstvie_kovra": [7, (255, 0, 128)],
    "8.Gribok/Mox": [8, (30, 30, 90)],
    "9.Rastreskivanie": [9, (255, 153, 204)],
    "10.Spolzanie_kovra": [10, (37, 193, 255)]
}

label_to_label = [
    ['0 - Фон', "background"],
    ['1 - Впадины', "1.Vpadini"],
    ['2 - Вздутие', "2.Vzdutie"],
    ['3 - Складка',"3.Skladki"],
    ['4 - Заплатки',"4.Zaplatki"],
    ['5 - Отслаивание ковра',"5.Otslaivanie_kovra"],
    ['6 - Разрывы ковра',"6.Razrivi_kovra"],
    ['7 - Отсутствие ковра',"7.Otsutstvie_kovra"],
    ['8 - Грибок/мох',"8.Gribok/Mox"],
    ['9 - Растрескивание',"9.Rastreskivanie"],
    ["10 - Сползание ковра","10.Spolzanie_kovra"]
]
mask_dict_2 = {
    '0 - Фон':[0,(255, 255, 255)],
    '1 - Впадины':[1,(0,0,128)],
    '2 - Вздутие':[2,(0,128,0)],
    '3 - Складка':[3,(255,165,0)],
    '4 - Заплатки':[4,(255,192,203)],
    '5 - Отслаивание ковра':[5,(255,0,0)],
    '6 - Разрывы ковра':[6,(179,157,219)],
    '7 - Отсутствие ковра':[7,(192,202,51)],
    '8 - Грибок/мох':[8,(3,155,229)],
    '9 - Растрескивание':[9,(121,85,72)],
    "10 - Сползание ковра": [10, (37, 193, 255)]}

def convert_mask_to_mask(path_from, path_to, mask_dict_1, mask_dict_2, label_to_label, inverse = False):
    file_list = os.listdir(path_from)
    os.makedirs(path_to,exist_ok=True)

    for count, filename in enumerate(sorted(file_list)):
        mask_path = os.path.join(path_from, filename)
        pil_image = Image.open(mask_path).convert('RGB')
        open_cv_image = np.array(pil_image)
        source_img = open_cv_image.copy()
        for pair in label_to_label:
            key_1 = pair[1]
            key_2 = pair[0]
            if inverse:
                color_2 = mask_dict_1[key_1][1]
                color_1 = mask_dict_2[key_2][1]
            else:
                color_1 = mask_dict_1[key_1][1]
                color_2 = mask_dict_2[key_2][1]

            tmp_img = np.copy(source_img)

            idx = cv2.inRange(tmp_img, np.array(color_1), np.array(color_1)).astype(np.bool)
            open_cv_image[idx]= np.array(color_2)

        pil_im = Image.fromarray(open_cv_image)
        pil_im.save(os.path.join(path_to, filename))
        print('Image '+ str(count)+': '+ filename)

# convert_mask_to_mask(
#     path_from = 'D:/Projects/Defects_Recognition/UNet_project/dataset/augmented_dataset/test/mask',
#     path_to = 'D:/Projects/Defects_Recognition/UNet_project/dataset/augmented_dataset/test/mask_labelbox',
#     mask_dict_1 = mask_dict_1,
#     mask_dict_2 = mask_dict_2,
#     label_to_label = label_to_label)

convert_mask_to_mask(
    path_from = 'D:/Projects/Defects_Recognition/UNet_project/data/membrane/train/validation_label',
    path_to = 'D:/Projects/Defects_Recognition/UNet_project/data/membrane/train/validation_label_net',
    mask_dict_1 = mask_dict_1,
    mask_dict_2 = mask_dict_2,
    label_to_label = label_to_label,
    inverse=True)

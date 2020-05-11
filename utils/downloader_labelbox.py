import json
import requests

img_path = 'train/image/'
mask_path = 'train/label/'

with open('export.json', 'r') as js:
    dataset = json.load(js)

for count, data in enumerate(dataset):
    if data['Label'] != 'Skip':
        img_url = data['Labeled Data']
        mask_url = data['Label']['segmentationMaskURL']

        img = open(img_path + str(count) + '.jpg', 'wb')
        ufr = requests.get(img_url)
        img.write(ufr.content)
        img.close()

        mask = open(mask_path + str(count) + '.png', 'wb')
        ufr = requests.get(mask_url)
        mask.write(ufr.content)
        mask.close()

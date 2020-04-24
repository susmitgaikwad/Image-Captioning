import numpy as np
import gist
import json
import scipy.io
from PIL import Image

dataset = json.load(open('../data/flickr8k/dataset.json'))
image_names = []
for image in dataset['images']:
    image_names.append(image['filename'])

print('number of images' + str(len(image_names)))

# create features data
feats = {'feats': []}

# extract features from images
for im_name in image_names:
    print(im_name)

    # read image
    img = Image.open('../data/Flicker8k_Dataset/'+im_name)

    # convert to array
    img = np.asarray(img)

    # get descriptor
    desc = gist.extract(img)

    # feats['feats'] = np.append(feats['feats'], flat)
    feats['feats'].append(desc.tolist())

scipy.io.savemat('../data/Flicker8k_KNN/knn_feats.mat', feats)

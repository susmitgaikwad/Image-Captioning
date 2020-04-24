from image_rnn import Image_LSTM
from feature_extraction.FeatureExtractor_Keras import feature_extraction_batch
import numpy as np

def captioning(images):
    
    # extract feature for all images
    feats = feature_extraction_batch(images)
    # initialize LSTM model
    model = Image_LSTM()
    model.load_weights('checkpoint_19.h5')
    model.set_dict()
    captions = []

    for feat in feats:
        caption = model.predict(feat)
        captions.append(caption)
        print(caption)
    return captions

if __name__ == '__main__':

    img1 = 'images/biker.jpg'
    img2 = 'images/dog-grass.jpg'
    img3 = 'images/dolphin.jpeg'
    img4 = 'images/man-run.jpg'
    img5 = 'images/splash_wave.png'
    img6 = 'images/woman-field.jpg'
    img7 = 'images/woman-field.png'
    

    
    imgs = [img1, img2, img3, img4, img5, img6, img7]

    captions = captioning(imgs)
    print(captions)

from keras.models import load_model
import cv2
import os
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

TEST_DIR = 'v/'

model = load_model('model.h5')

def normalize(x, mean, std):
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    x[..., 0] /= std[0]
    x[..., 1] /= std[1]
    x[..., 2] /= std[2]
    return x

for filename in os.listdir(r'v/'):
    if filename.endswith(".jpg") or filename.endswith(".ppm") or filename.endswith(".jpeg") or filename.endswith(".png"):
        ImageCV = cv2.resize(cv2.imread(os.path.join(TEST_DIR) + filename), (224,224))
        #image_blurred = cv2.GaussianBlur(ImageCV, (0, 0), 100 / 30)
        #ImageCV = cv2.addWeighted(ImageCV, 4, image_blurred, -4, 128)
        #ImageCV = ImageCV.reshape(-1,224,224,3)
        
        x = image.img_to_array(ImageCV)
        x = np.expand_dims(x, axis=0)
        x = normalize(x, [59.5105,61.141457,61.141457], [60.26705,61.85445,63.139835])
        #x = preprocess_input(x)
        
        prob = model.predict(x)
        if prob <= 0.5:
            print("nonPDR >>>", filename)
        else:
            print("PDR >>>", filename)
        #cv2.imshow('image', ImageCV)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
#
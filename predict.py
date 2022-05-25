from keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

model = load_model("final.h")
img_dims = 150
path = './chest_xray/'

test_val_datagen = ImageDataGenerator(
            rescale=1. / 255
        )
val_gen = test_val_datagen.flow_from_directory(
        directory=path + 'pre',
        target_size=(img_dims, img_dims),
        batch_size=1,
        class_mode='binary',
        shuffle=False)

"""model.summary()

print('##################')

print(model.layers)
print('num of layers: ' + str(len(model.layers)) + ' layer')"""


# path = './chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg'
# img = plt.imread(path)
# img = cv2.resize(img, (150, 150))
# img = np.dstack([img, img, img])
# img = img.astype('float32') / 255
# pre = model.predict(img)
# print(pre)

# model.predict('\chest_xray\val\PNEUMONIA\person1946_bacteria_4874.jpeg')

preds = model.predict(val_gen);
for i in preds :
        if(i <= .5):
                print("Normal")
        else :
                print("Not Normal")

import tensorflow as tf
import numpy as np
import cv2


def img_processing(path):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)   
    return img
num_to_word = {0:'子', 1:'丑', 2:'寅', 3:'卯', 4:'辰', 5:'巳', 6:'午', 7:'未', 8:'申', 9:'酉', 10:'戌', 11:'亥', 12:' '}


model = tf.keras.models.load_model("models/12class_v4_simple_128.h5")


cap=cv2.VideoCapture(0)
while(1):
    ret ,frame = cap.read()
    # print(ret)
    frame=cv2.flip(frame,1)
    color = (0, 255, 255)
    
    cv2.rectangle(frame, (190, 210), (449, 469), color, 2)
    k=cv2.waitKey(1)

    # print(frame[212:468,192:448].shape) 256 256 3
    grayImage = cv2.cvtColor(frame[212:468,192:448], cv2.COLOR_BGR2GRAY)
    grayImage = cv2.resize(grayImage, (128, 128), interpolation=cv2.INTER_AREA)
    grayImage = np.expand_dims(grayImage, axis=-1)
    x = np.expand_dims(grayImage, axis=0)
    # print(x.shape) 1 256 256 1

    
    
    # print(my_tensor)
    y_pred = model.predict_classes(x)
    print(num_to_word[y_pred[0]])
    cv2.imshow("capture", grayImage)
    
cap.release()
cv2.destroyAllWindows()
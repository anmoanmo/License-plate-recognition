import cv2
import numpy as np
from tensorflow import keras




def cnn_predict(cnn, Lic_img):
    characters = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
                  "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2",
                  "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
                  "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    Lic_pred = []
    if Lic_img is None:
        return Lic_pred
    for lic in Lic_img:
        resized_lic = cv2.resize(lic, (240, 80)).astype("float32") / 255.0
        lic_pred = cnn.predict(resized_lic.reshape(1, 80, 240, 3), verbose=0)
        lic_pred = np.array(lic_pred).reshape(7, 65)
        if len(lic_pred[lic_pred >= 0.8]) >= 4:
            chars = ''
            for arg in np.argmax(lic_pred, axis=1):
                chars += characters[arg]
            chars = chars[0:2] + '·' + chars[2:]
            Lic_pred.append((lic, chars))
    return Lic_pred




#
# if __name__ == '__main__':
#
#
#      cnn = keras.models.load_model('cnn.h5')
#      coordinates = cnn_predict(cnn,'test_images/test.jpg')
#      print(coordinates)






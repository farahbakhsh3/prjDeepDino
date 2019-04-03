import os
import datetime
import keyboard
import numpy as np
from time import sleep
from PIL import ImageGrab
from skimage.io import imsave
from Prepare import model_def, Sample

class Key:
    def __init__(self):
        self.counter = 0
        self.i = 0
key = Key()

def datatime_fldr_name():
    now = datetime.datetime.now()
    y = str(now.year)
    m = str(now.month)
    if len(m) == 1: m = '0' + m
    d = str(now.day)
    if len(d) == 1: d = '0' + d
    h = str(now.hour)
    if len(h) == 1: h = '0' + h
    mi = str(now.minute)
    if len(mi) == 1: mi = '0' + mi
    s = str(now.second)
    if len(s) == 1: s = '0' + s
    return y + m + d + '_' + h + mi + s + '/'

def key_pressed():
    print('key_pressed : Start')
    while 1:
        if keyboard.is_pressed('up'):
            capture1('1.U')
        elif keyboard.is_pressed('down'):
            capture1('2.D')
        elif keyboard.is_pressed('ctrl+c'):
            break
        else:
            if key.counter >= 1000:
                capture1('0.N')
                key.counter = 0
            else:
                key.counter += 1

def capture1(_key):
    screen = ImageGrab.grab(bbox=(Sample.IMG_Crop_Y[0],
                                  Sample.IMG_Crop_X[0],
                                  Sample.IMG_Crop_Y[1],
                                  Sample.IMG_Crop_X[1]))
    key.i += 1
    fName = img_folders_path + str2(key.i) + '.' + _key + '.jpg'
    print(fName, ': saved.')
    screen.save(fName, format='JPEG')

def str2(i):
    if i < 10:
        return '00000' + str(i)
    elif i < 100:
        return '0000' + str(i)
    elif i < 1000:
        return '000' + str(i)
    elif i < 10000:
        return '00' + str(i)
    elif i < 100000:
        return '0' + str(i)
    else:
        return str(i)

if __name__ == '__main__':
    m_model_def = model_def()
    data_folders_path = m_model_def.data_fldr
    if not os.path.exists(data_folders_path):
        os.mkdir(data_folders_path)
        print(data_folders_path, 'created.')
    img_folders_path = data_folders_path + datatime_fldr_name()
    if not os.path.exists(img_folders_path):
        os.mkdir(img_folders_path)
        print(img_folders_path, 'created.')

    key_pressed()
    print('-----------------')
    print('Process finished.')
    print(key.i, 'images saved in ', img_folders_path)


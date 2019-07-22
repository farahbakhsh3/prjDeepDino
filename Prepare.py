import os
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize


class Sample:
    IMG_W = 320
    IMG_H = 180
    IMG_D = 1

    # ▲ X
    # │
    # │
    # └───────► Y
    # IMG_Crop_X = (x1, x2) #(from, to)
    # IMG_Crop_Y = (y1, y2) #(from, to)
    IMG_Crop_X = (200, 700)
    IMG_Crop_Y = (10, 1580)


class ModelDef:
    def __init__(self):
        self.random_seed = 7
        self.game = 'Dino'
        self.INPUT_SHAPE = (Sample.IMG_H, Sample.IMG_W, Sample.IMG_D)
        self.OUT_SHAPE = 3
        self.weights_file = self.game + '.h5'
        self.data_fldr = './Data/'
        self.need_crop = False

        print('---------------------------')
        print('Random seed: ', self.random_seed)
        print('Game: ', self.game, ' Selected.')
        print('INPUT SHAPE: ', self.INPUT_SHAPE)
        print('OUT SHAPE: ', self.OUT_SHAPE)
        print('Weights file: ', self.weights_file)
        print('Data folder: ', self.data_fldr)
        print('Need crop images: ', self.need_crop)
        print('---------------------------')
        print()


def resize_image(img):
    # img = img[Sample.IMG_Crop_X[0]:Sample.IMG_Crop_X[1], Sample.IMG_Crop_Y[0]:Sample.IMG_Crop_Y[1]]
    img = resize(img, (Sample.IMG_H, Sample.IMG_W), mode='reflect', anti_aliasing=True)
    img = rgb2gray(img)
    img = img.astype('float16')
    img = img.reshape((Sample.IMG_H, Sample.IMG_W, Sample.IMG_D))
    return img


def prepare():
    X = []
    Y = []

    data_folder_path = m_model_def.data_fldr
    folders = os.listdir(data_folder_path)
    print('Total Folders: ', len(folders))
    try:
        for folder in folders:
            print('Folder: ', folder)
            pics = os.listdir(data_folder_path + folder)
            for pic in tqdm(pics):
                image = imread(data_folder_path + folder + '/' + pic)
                vec = resize_image(image)
                X.append(vec)
                Y.append(pic.split('.')[1])
    except KeyboardInterrupt:
        pass

    X = np.asarray(X)
    Y = np.asarray(Y)

    print('---------------------------')
    print('X.Shape=', X.shape)
    print('Y.Shape=', Y.shape)
    print('---------------------------')

    np.save('X.npy', X)
    np.save('Y.npy', Y)


if __name__ == '__main__':
    m_model_def = ModelDef()
    prepare()

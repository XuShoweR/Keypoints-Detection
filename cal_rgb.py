import os
import cv2
import numpy as np

path = '/media/roots/keypoints_detection/Datasets/train/Images'


def compute(path):
    dir_names = os.listdir(path)
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []

    for dir_name in dir_names:
        dir_name = os.path.join(path, dir_name)
        file_names = os.listdir(dir_name)
        for file_name in file_names:
            img = cv2.imread(os.path.join(dir_name, file_name), 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            per_image_Rmean.append(np.mean(img[:, :, 0]))
            per_image_Gmean.append(np.mean(img[:, :, 1]))
            per_image_Bmean.append(np.mean(img[:, :, 2]))
    R_mean = np.mean(per_image_Rmean)
    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)
    return R_mean, G_mean, B_mean


if __name__ == '__main__':
    R, G, B = compute(path)
    print(R, G, B)

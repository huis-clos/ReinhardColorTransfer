import cv2
import numpy as np
import matplotlib.pyplot as plt
import math as m
import cv


def lab_comp():
    img_lab_source = cv2.imread('img1.png')

    img_lab_target = cv2.imread('img2.png')

    img_lab_source = cv2.cvtColor(img_lab_source, cv2.COLOR_BGR2RGB)

    img_lab_target = cv2.cvtColor(img_lab_target, cv2.COLOR_BGR2RGB)

    img_lab_source = cv2.cvtColor(img_lab_source, cv2.COLOR_RGB2XYZ)

    img_lab_target = cv2.cvtColor(img_lab_target, cv2.COLOR_RGB2XYZ)

    # normalize image if not already in correct format
    if img_lab_source.dtype != 'float32':
        img_source = cv2.normalize(img_lab_source.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
#
    if img_lab_target.dtype != 'float32':
        img_lab_target = cv2.normalize(img_lab_target.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)


    # normalize image if not already in correct format

    # get dimensions of input image
    x_s, y_s, z_s = img_lab_source.shape

    x_t, y_t, z_t = img_lab_target.shape

    # Matrices for conversion from paper
    LMS2LAB = np.matrix(([(1 / m.sqrt(3), 0, 0), (0, 1 / m.sqrt(6), 0), (0, 0, 1 / m.sqrt(2))]))
    ident = np.matrix(([(1, 1, 1,), (1, 1, -2), (1, -1, 0)]))

    ident2 = np.matrix(([(1, 1, 1), (1, 1, -1), (1, -2, 0)]))

    LAB2LMS_final = ident2 * LMS2LAB

    # Get standard deviation and mean from channel in
    # source image and target image
    mean_sl = np.matrix(img_lab_source[:, :, 0]).mean()
    mean_sa = np.matrix(img_lab_source[:, :, 1]).mean()
    mean_sb = np.matrix(img_lab_source[:, :, 2]).mean()

    std_sl = np.matrix(img_lab_source[:, :, 0]).std()
    std_sa = np.matrix(img_lab_source[:, :, 1]).std()
    std_sb = np.matrix(img_lab_source[:, :, 2]).std()

    mean_tl = np.matrix(img_lab_target[:, :, 0]).mean()
    mean_ta = np.matrix(img_lab_target[:, :, 1]).mean()
    mean_tb = np.matrix(img_lab_target[:, :, 2]).mean()

    std_tl = np.matrix(img_lab_target[:, :, 0]).std()
    std_ta = np.matrix(img_lab_target[:, :, 1]).std()
    std_tb = np.matrix(img_lab_target[:, :, 2]).std()

    std_l = std_tl / std_sl
    std_a = std_ta / std_sa
    std_b = std_tb / std_sb

    res_lab = np.zeros((x_s, y_s, z_s), 'float32')

    for i in range(x_s):
        for j in range(y_s):
            res_lab[i][j][0] = mean_tl + std_l * (img_lab_source[i][j][0] - mean_sl)
            res_lab[i][j][1] = mean_ta + std_a * (img_lab_source[i][j][1] - mean_sa)
            res_lab[i][j][2] = mean_tb + std_b * (img_lab_source[i][j][2] - mean_sb)

    lab_img = cv2.cvtColor(res_lab, cv2.COLOR_XYZ2RGB)

    lab_img[:, :, 0] = np.clip(res_lab[:, :, 0], 0, 1)
    lab_img[:, :, 1] = np.clip(res_lab[:, :, 1], 0, 1)
    lab_img[:, :, 2] = np.clip(res_lab[:, :, 2], 0, 1)

    plt.imshow(lab_img)
    plt.show()

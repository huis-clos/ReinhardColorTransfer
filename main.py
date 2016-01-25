#!/usr/bin/python

import skimage.color as color
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math as m
import matplotlib.cm as cm
import scipy

img2 = cv2.imread('scotland_house.jpg', flags=cv2.IMREAD_COLOR)

img1 = cv2.imread('scotland_plain.jpg', flags=cv2.IMREAD_COLOR)

img_source = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

img_target = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)

# normalize image if not already in correct format
if img_source.dtype != 'float32':
    img_source = cv2.normalize(img_source.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)

if img_target.dtype != 'float32':
    img_target = cv2.normalize(img_target.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)

# get dimensions of input image
x_s, y_s, z_s = img_source.shape

x_t, y_t, z_t = img_target.shape

R = img_source[:, :, 0]
G = img_source[:, :, 1]
B = img_source[:, :, 2]

RGB2LMS = np.asarray(([(0.3811, 0.5783, 0.0402), (0.1967, 0.7244, 0.0782), (0.0241, 0.1288, 0.8444)]))
LMS2LAB = np.asarray(([(1 / m.sqrt(3), 0, 0), (0, 1 / m.sqrt(6), 0), (0, 0, 1 / m.sqrt(2))]))
ident = np.asarray(([(1, 1, 1,), (1, 1, -2), (1, -1, 0)]))
LAB2LMS = np.asarray(([(m.sqrt(3) / 3, 0, 0), (0, m.sqrt(6) / 6, 0), (0, 0, m.sqrt(2) / 2)]))
ident2 = np.asarray(([(1, 1, 1), (1, 1, -1), (1, -2, 0)]))
LMS2LAB_final = LMS2LAB * ident
LAB2LMS_final = LAB2LMS * ident2

LMS2RGB = np.asarray(([(4.4679, -3.5873, 0.1193), (-1.2186, 2.3809, -0.1624), (0.0497, -0.2439, 1.2045)]))

LMS_s = np.zeros((x_s, y_s, z_s), 'float32')

LMS_t = np.zeros((x_s, y_s, z_s), 'float32')

for i in range(x_s):
    for j in range(y_s):
        k = img_source[i][j][np.newaxis, :]
        LMS_s[i][j] = np.dot(k, RGB2LMS)
        LMS_s[i][j] = np.log10(LMS_s[i][j])

for i in range(x_t):
    for j in range(y_t):
        k = img_target[i][j][np.newaxis, :]
        LMS_t[i][j] = np.dot(k, RGB2LMS)
        LMS_t[i][j] = np.log10(LMS_t[i][j])

lab_s = np.zeros((x_s, y_s, z_s), 'float32')

lab_t = np.zeros((x_s, y_s, z_s), 'float32')

for i in range(x_s):
    for j in range(y_s):
        k = img_target[i][j][np.newaxis, :]
        lab_s[i][j] = np.dot(k, LMS2LAB_final)

for i in range(x_t):
    for j in range(y_t):
        k = img_target[i][j][np.newaxis, :]
        lab_t[i][j] = np.dot(k, LMS2LAB_final)

mean_sl = np.mean(lab_s[:, :, 0])
mean_sa = np.mean(lab_s[:, :, 1])
mean_sb = np.mean(lab_s[:, :, 2])
std_sl = np.std(lab_s[:, :, 0])
std_sa = np.std(lab_s[:, :, 1])
std_sb = np.std(lab_s[:, :, 2])
mean_tl = np.mean(lab_t[:, :, 0])
mean_ta = np.mean(lab_t[:, :, 1])
mean_tb = np.mean(lab_t[:, :, 2])
std_tl = np.std(lab_t[:, :, 0])
std_ta = np.std(lab_t[:, :, 0])
std_tb = np.std(lab_t[:, :, 0])

std_l = np.divide(std_sl, std_tl)
std_a = np.divide(std_sa, std_ta)
std_b = np.divide(std_sb, std_tb)

res_lab = np.zeros((x_s, y_s, z_s), 'float32')

for i in range(x_s):
    for j in range(y_s):
        res_lab[i][j][0] = mean_tl + std_l * lab_s[i][j][0] - mean_sl
        res_lab[i][j][1] = mean_ta + std_a * lab_s[i][j][1] - mean_sa
        res_lab[i][j][2] = mean_tb + std_b * lab_s[i][j][2] - mean_sb

LMS_res = np.zeros((x_s, y_s, z_s), 'float32')

for i in range(x_s):
    for j in range(y_s):
        k = res_lab[i][j][np.newaxis, :]
        LMS_res[i][j] = np.dot(k, LAB2LMS_final)
        LMS_res[i][j] = np.power(10, LMS_res[i][j])

est_im = np.zeros((x_s, y_s, z_s), 'float32')

for i in range(x_s):
    for j in range(y_s):
        k = LMS_res[i][j][np.newaxis, :]
        est_im[i][j] = np.dot(k, LMS2RGB)

est_im = np.reshape(est_im, img_source.shape)

plt.imshow(est_im)
plt.show()

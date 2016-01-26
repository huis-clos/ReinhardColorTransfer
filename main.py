#!/usr/bin/python
# Colleen Toth
# Project 1, Intro to Computer Vision
# Prof. Feng Liu
#
# Python Implementation of 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math as m


img_source = cv2.imread('img1.png')

img_target = cv2.imread('img2.png')

img_source = cv2.cvtColor(img_source, cv2.COLOR_RGB2BGR)

img_target = cv2.cvtColor(img_target, cv2.COLOR_RGB2BGR)

# normalize image if not already in correct format
if img_source.dtype != 'float64':
   img_source = cv2.normalize(img_source.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
#
if img_target.dtype != 'float64':
   img_target = cv2.normalize(img_target.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)

# get dimensions of input image
x_s, y_s, z_s = img_source.shape

x_t, y_t, z_t = img_target.shape

# Matrices for conversion from paper
RGB2LMS = np.matrix(([(0.3811, 0.5783, 0.0402), (0.1967, 0.7244, 0.0782), (0.0241, 0.1288, 0.8444)]))
LMS2LAB = np.matrix(([(1 / m.sqrt(3), 0, 0), (0, 1 / m.sqrt(6), 0), (0, 0, 1 / m.sqrt(2))]))
ident = np.matrix(([(1, 1, 1,), (1, 1, -2), (1, -1, 0)]))

LAB2LMS = np.matrix(([(m.sqrt(3) / 3, 0, 0), (0, m.sqrt(6) / 6, 0), (0, 0, m.sqrt(2) / 2)]))
ident2 = np.matrix(([(1, 1, 1), (1, 1, -1), (1, -2, 0)]))

LMS2LAB_final = LMS2LAB * ident
LAB2LMS_final = ident2 * LMS2LAB

LMS2RGB = np.matrix(([(4.4679, -3.5873, 0.1193), (-1.2186, 2.3809, -0.1624), (0.0497, -0.2439, 1.2045)]))

LMS_s = np.zeros((x_s, y_s, z_s), 'float64')

LMS_t = np.zeros((x_t, y_t, z_t), 'float64')

# Conversion from RGB to LMS for source
for i in range(x_s):
    for j in range(y_s):
        k = np.matrix(img_source[i][j][np.newaxis, :]).T
        temp = RGB2LMS * k
        LMS_s[i][j][0], LMS_s[i][j][1], LMS_s[i][j][2] = np.log10(temp)

# Conversion from RGB to LMS for
for i in range(x_t):
    for j in range(y_t):
        l = np.matrix(img_target[i][j][np.newaxis, :]).T
        temp1 = RGB2LMS * l
        LMS_t[i][j][0], LMS_t[i][j][1], LMS_t[i][j][2] = np.log10(temp1)

lab_s = np.zeros((x_s, y_s, z_s), 'float64')

lab_t = np.zeros((x_t, y_t, z_t), 'float64')

# Convert LMS to LAB
for i in range(x_s):
    for j in range(y_s):
        n = np.matrix(LMS_s[i][j][np.newaxis, :]).T
        lab_s[i][j][0], lab_s[i][j][1], lab_s[i][j][2] = LMS2LAB_final * n

for i in range(x_t):
    for j in range(y_t):
        o = np.matrix(LMS_t[i][j][np.newaxis, :]).T
        lab_t[i][j][0], lab_t[i][j][1], lab_t[i][j][2] = LMS2LAB_final * o

R = np.matrix(lab_t[:, :, 0])
sl = R.mean()

mean_sl = np.matrix(lab_s[:, :, 0]).mean()
mean_sa = np.matrix(lab_s[:, :, 1]).mean()
mean_sb = np.matrix(lab_s[:, :, 2]).mean()

std_sl = np.matrix(lab_s[:, :, 0]).std()
std_sa = np.matrix(lab_s[:, :, 1]).std()
std_sb = np.matrix(lab_s[:, :, 2]).std()

mean_tl = np.matrix(lab_t[:, :, 0]).mean()
mean_ta = np.matrix(lab_t[:, :, 1]).mean()
mean_tb = np.matrix(lab_t[:, :, 2]).mean()

std_tl = np.matrix(lab_t[:, :, 0]).std()
std_ta = np.matrix(lab_t[:, :, 1]).std()
std_tb = np.matrix(lab_t[:, :, 2]).std()

std_l = std_tl / std_sl
std_a = std_ta / std_sa
std_b = std_tb / std_sb

res_lab = np.zeros((x_s, y_s, z_s), 'float64')

for i in range(x_s):
    for j in range(y_s):
        res_lab[i][j][0] = mean_tl + std_l * (lab_s[i][j][0] - mean_sl)
        res_lab[i][j][1] = mean_ta + std_a * (lab_s[i][j][1] - mean_sa)
        res_lab[i][j][2] = mean_tb + std_b * (lab_s[i][j][2] - mean_sb)

LMS_res = np.zeros((x_s, y_s, z_s), 'float64')

for i in range(x_s):
    for j in range(y_s):
        p = np.matrix(res_lab[i][j][np.newaxis, :]).T
        temp3 = LAB2LMS_final * p
        LMS_res[i][j][0], LMS_res[i][j][1], LMS_res[i][j][2] = np.power(10, temp3)

est_im = np.zeros((x_s, y_s, z_s), 'float64')

for i in range(x_s):
    for j in range(y_s):
        q = np.matrix(LMS_res[i][j][np.newaxis, :]).T
        est_im[i][j][0], est_im[i][j][1], est_im[i][j][2] = LMS2RGB * q

est_im[:, :, 0] = np.clip(est_im[:, :, 0], 0, 1)
est_im[:, :, 1] = np.clip(est_im[:, :, 1], 0, 1)
est_im[:, :, 2] = np.clip(est_im[:, :, 2], 0, 1)

plt.imshow(est_im)
plt.show()


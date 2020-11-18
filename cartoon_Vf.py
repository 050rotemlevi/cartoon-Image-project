import cv2
import numpy as np
from cartoon import cartooneff
from cartoon import FindPeopleDNN
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt

## import image
image = cv2.imread("123.jpeg")

## find people and create mask
treshold = 0.2
FindPeopleDNN.segment_image_dnn(image, treshold)

## import mask
mask = cv2.imread("mask_with_segm_num.png")

## create subImages
OnlyBack = image * (1- mask//255)
OnlyPeople = image * (mask//255)

## cartoon background
backCartoon = cartooneff.cartoon_eff(OnlyBack)

## merge people and background
BackCartoonAndpeople = backCartoon + OnlyPeople
cv2.imshow("BackCartoonAndpeople", BackCartoonAndpeople)
cv2.imwrite("cartoon_pic/BackCartoonAndpeople.png",BackCartoonAndpeople)

cv2.waitKey(0)
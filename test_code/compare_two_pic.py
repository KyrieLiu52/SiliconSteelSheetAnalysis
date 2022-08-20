import cv2
import numpy as np

# pic1 = "./170C-24H-8000x.hogmag.png"
# pic2 = "./python_hogmag.png"

pic1 = "./170C-24H-8000x.hogdirt.png"
pic2 = "./python_hogdirt.png"

src1 = cv2.imread(pic1, flags=cv2.IMREAD_COLOR)
img1 = cv2.cvtColor(src1, code=cv2.COLOR_BGR2GRAY, dstCn=0)

src2 = cv2.imread(pic2, flags=cv2.IMREAD_COLOR)
img2 = cv2.cvtColor(src2, code=cv2.COLOR_BGR2GRAY, dstCn=0)

arr_img_1 = np.asarray(img1)
arr_img_2 = np.asarray(img2)

print(arr_img_1.shape)
print(arr_img_2.shape)

print(np.max(np.subtract(arr_img_1, arr_img_2)))
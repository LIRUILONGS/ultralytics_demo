#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   service.py
@Time    :   2023/05/22 12:13:32
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   
"""

# here put the import lib

import cv2
import imutils
from deepface import DeepFace
from decimal import Decimal
from imutils import paths
import os
import units 
import copy



def extract_faces_all(image, detector_backend='retinaface', align=True):
    """
    @Time    :   2023/05/20 03:50:07
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   extract_faces 用于对图像进行特征分析，提取头像坐标，
                    在实际使用中，如果对精度有要求，可以通过 `confidence` 来对提取的人脸进行过滤，
                 Args:
                 extract_faces方法接受以下参数：
                    - img_path：要从中提取人脸的图像路径、numpy数组（BGR）或base64编码的图像。
                    - target_size：人脸图像的最终形状。将添加黑色像素以调整图像大小。
                    - detector_backend：人脸检测后端可以是retinaface、mtcnn、opencv、ssd或dlib。
                    - enforce_detection：如果在提供的图像中无法检测到人脸，则该函数会引发异常。如果不想得到异常并仍要运行该函数，则将其设置为False。
                    - align：根据眼睛位置对齐。
                    - grayscale：以RGB或灰度提取人脸。

                 Returns:
                   返回一个包含人脸图像、人脸区域和置信度的字典列表。其中，
                   - face 键对应的值是提取的人脸图像
                   - facial_area 键对应的值是人脸在原始图像中的位置和大小
                   - confidence 键对应的值是人脸检测的置信度

    """
    # img_path = "huge_1.jpg"

    # 读取原始图像
    rst = None
    try:
        rst = DeepFace.extract_faces(
            img_path=image,
            target_size=(224, 224),
            detector_backend="mtcnn",
            enforce_detection=True,
            align=True,
            grayscale=False)
    except Exception as e:
        print(e)
        return

    # print(rst)
    # 人脸坐标和置信度
    to_deepfaces = []
    image_no_mark = copy.deepcopy(image)
    for i, f in enumerate(rst):
        # print(i, f)
        #         print('😊'.rjust(i * 2, '😊'))
        print("编号：", i, '\n', " 检测人脸位置:", f['facial_area'], '\n', " 置信度:", f['confidence'])
        x, y, w, h = f['facial_area'].values()
        x1, y1, x2, y2 = x, y, x + w, y + h
        # 根据不同的置信度做不同标记
        confidence = Decimal(str(f['confidence']))
        best = Decimal('1')
        color = (0, 255, 0)

        abs_ = best - confidence
        if 0.01 <= abs_ < 0.05:
            color = (255, 0, 255)
        elif 0.08 > abs_ >= 0.05:
            color = (0, 165, 255)
        elif abs_ >= 0.08:
            color = (255, 255, 255)
        else:
            pass
            # 这个精度的考虑用于识别
        cropped_img = image_no_mark[y:y + h, x:x + w]
         #对切片进行等比放大
        cropped_img = imutils.resize(cropped_img, width=400)
        #cv2.imwrite('new_' + str(i) + img_path, cropped_img)
        img_c =  units.get_img_to_base64(cropped_img)
        to_deepfaces.append({"facial_area":f['facial_area'],"confidence":format(f['confidence'], '0.4f'),"img_b64":img_c})

        # 根据坐标标记图片,标记框的左上角和右下角的坐标,
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        # 添加 置信度标签
        cv2.putText(image, format(f['confidence'], '0.4f'), (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2,
                    cv2.LINE_4)
    return image,rst,to_deepfaces
        


if __name__ == "__main__":
    for fn in  paths.list_images("./images"):
        image,_,_ = extract_faces_all(fn, detector_backend='retinaface', align=True)
        fn = os.path.basename(fn)
        ip = str(fn).split("_01_")[0]
        cv2.imwrite("./"+ip+ "/"+ip + "_dc.jpg",image)
        
"""
人脸检测模块(Haar 级联分类器)
职责: 定位定位人脸位置, 返回人脸框坐标[(x, y, w, h), ...]
"""

import os
import cv2
import numpy as np


class FaceDetection:

    def __init__(self, model_name = 'haarcascade_frontalface_alt2.xml'):
        """
        加载 OpenCV 自带的 Haar 预训练模型
        """
        if not hasattr(cv2, 'CascadeClassifier'):
            raise RuntimeError(
                "当前 opencv-python 版本不支持 Haar 级联检测."
            )
        self.face_engine = cv2.CascadeClassifier(cv2.data.haarcascades + model_name)
        if self.face_engine.empty():
            raise ValueError(f"无法加载人脸检测模型: {model_name}")


    # --------------------------------------- 图像预处理 ---------------------------------------
    @staticmethod
    def preprocess(image):
        """
        图像预处理: 消除光照与噪声, 降低算法计算压力
        """

        # 1. 色彩空间转换: BGR -> 灰度图(Haar 检测只需要的灰度信息)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 2. 光照均衡化: 直方图均衡化, 解决明暗差异过大问题
        gray = cv2.equalizeHist(gray)
        # 3. 图像降噪: 高斯模糊去除画面噪点
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        return gray


    # --------------------------------------- 人脸检测 ---------------------------------------
    def detect_faces(self, image, scale_factor = 1.1, min_neighbors = 5, min_size = (50, 50)):
        """
        检测图中人脸
        :param image: BGR 图像
        :return: 人脸坐标列表[(x, y, w, h), ...]
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("图像数据类型错误: 应为numpy.ndarray")

        gray = self.preprocess(image)
        faces = self.face_engine.detectMultiScale(
            gray,
            scaleFactor = scale_factor,
            minNeighbors = min_neighbors,
            minSize = min_size
        )

        # 边界校验: 过滤越界或过小的框, 防止后续切片报错
        height, width = image.shape[:2]
        valid_faces = []
        for (x, y, w, h) in faces:
            if w < 10 or h < 10:
                continue                # 框过小, 忽略
            x2, y2 = x + w, y + h
            if x < 0 or y < 0 or x2 > width or y2 > height:
                continue                # 框越界, 忽略
            valid_faces.append((x, y, w, h))
        return valid_faces


    # --------------------------------------- ROI 截取 ---------------------------------------
    @staticmethod
    def crop_face_rois(image, faces):
        """
        根据人脸框坐标, 从原图切割提取每一张人脸子图像
        :return: [(roi图像, 坐标), ...]
        """

        rois = []
        for (x, y, w, h) in faces:
            rois.append((image[y: y + h, x: x + w], (x, y, w, h)))
        return rois
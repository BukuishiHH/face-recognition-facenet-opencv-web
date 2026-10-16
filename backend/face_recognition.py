"""
人脸识别模块(FaceNet 深度学习特征比对)
职责: 提取人脸 512 维特征向量, 与特征库比对判断"这个人是谁"
"""

import os
import cv2
import torch
import numpy as np
from db_utils import face_db
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity


class FaceRecognition:


    # 类变量: 模型实例全局复用, 避免重复加载
    _model = None
    _device = None


    # --------------------------------------- 模型加载 ---------------------------------------
    @classmethod
    def _get_model(cls):
        if cls._model is None:
            cls._device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._model = InceptionResnetV1(pretrained='casia-webface').eval().to(cls._device)
            print(f"FaceNet 模型加载完成(设备：{cls._device})")
        return cls._model


    # --------------------------------------- 特征提取 ---------------------------------------
    @classmethod
    def extract_feature(cls, face_img):
        """
        提取人脸特征
        :return: 归一化后的 512 维特, 形状(1, 512)
        """
        model = cls._get_model()

        # 1. 若传入路径则读取
        if isinstance(face_img, str):
            if not os.path.exists(face_img):
                print(f"Error! 图像文件不存在：{face_img}")
                return None
            face_img = cv2.imdecode(np.fromfile(face_img, dtype=np.uint8), cv2.IMREAD_COLOR)
            if face_img is None:
                print(f"Error! 无法读取图像：{face_img}")
                return None

        try:
            # 2. 预处理: BGR -> RGB(深度学习模型需要), 缩放到模型输入尺寸
            img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (160, 160))

            # 3. 转张量归一化, 与模型设备保持一致
            face_tensor = (torch.tensor(img_rgb).permute(2, 0, 1).float() / 255.0).unsqueeze(0)
            face_tensor = face_tensor.to(cls._device)

            # 4. 提取特征并做 L2 归一化
            with torch.no_grad():
                embedding = model(face_tensor)
                l2_norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
                normalized_feature = (embedding / l2_norm).cpu().numpy()

            return normalized_feature

        except Exception as e:
            print(f"Error! 人脸特征提取失败：{e}")
            return None


    # --------------------------------------- 身份对比 ---------------------------------------
    @classmethod
    def recognize_face(cls, face_image, threshold=0.6):
        """
        识别单张人脸: 提取特征 -> 与特征库所有特征算余弦相似度 -> 取最高分
        :param threshold: 相似度阈值, 大于等于阈值判定为同一个人
        :return: {'name': 姓名, 'similarity': 相似度} 或 None(无匹配)
        """

        # 1. 提取待识别的人脸特征
        target_feature = cls.extract_feature(face_image)
        if target_feature is None:
            return None

        # 2. 获取特征库所有特征
        registered_features, info_list = face_db.get_all_features()
        if registered_features is None or len(registered_features) == 0:
            print("Warning! 特征库为空, 请先录入人脸")
            return None

        # 3. 计算余弦相似度, 值越大越相似, 取最高分
        similarities = cosine_similarity(target_feature, registered_features)[0]
        max_idx = int(np.argmax(similarities))
        max_similarity = float(similarities[max_idx])

        # 4. 阈值判断
        if max_similarity >= threshold:
            return {'name': info_list[max_idx]['name'], 'similarity': max_similarity}
        print(f"Error! 无匹配人脸(最高相似度 {max_similarity:.2f} < 阈值 {threshold})")
        return None
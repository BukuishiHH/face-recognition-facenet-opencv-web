"""
人脸检测识别系统- 主程序
业务流程(每步对应函数):
  1. 图像源获取(read_image / 摄像头循环)
  2. 图像预处理(FaceDetection.preprocess: 灰度、均衡化、降噪)
  3. 人脸检测 + ROI截取(FaceDetection.detect_faces / crop_face_rois)
  4. 人脸识别(FaceRecognition: FaceNet提特征 + 特征库比对)
  5. 结果可视化与输出(draw_results / save_result)
  6. 资源释放(cap.release + destroyAllWindows)
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from db_utils import face_db
from face_detection import FaceDetection
from face_recognition import FaceRecognition

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, 'result_output')
face_detector = FaceDetection(model_name='haarcascade_frontalface_alt2.xml')


# --------------------------------------- 通用工具 ---------------------------------------
def read_image(image_path):
    """
    读取本地图片, 校验合法性
    """
    if not os.path.exists(image_path):
        print(f"Error! 图片文件不存在: {image_path}")
        return None
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error! 无法读取图片: {image_path}(可能格式不支持或文件损坏)")
    return image

def put_chinese_text(img, text, position, font_size=22, color=(0, 255, 0)):
    """
    在 OpenCV 图像中绘制中文
    """
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", font_size)
    except Exception as e:
        font = ImageFont.load_default()
        print("Warning! 中文字体加载失败, 可能显示乱码")
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_results(image, faces, results):
    """
    在原图上绘制人脸框 + 标注身份和置信度
    :param faces: [(x, y, w, h), ...]
    :param result: 识别结果列表, 与 faces 一一对应, 无匹配处为 None
    """
    for (x, y, w, h), result in zip(faces, results):
        if result is not None:
            text = f"{result['name']} ({result['similarity']:.2f})"
            color = (0, 255, 0)          # 识别成功: 绿框绿字
        else:
            text = "Unknown"
            color = (0, 0, 255)          # 无匹配: 红框红字
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        image = put_chinese_text(image, text, (x, y - 30), font_size=20, color=color)
    return image

def save_result(image, prefix='result'):
    """
    保存标注后的结果图
    """
    os.makedirs(RESULT_DIR, exist_ok=True)
    save_path = os.path.join(RESULT_DIR, f"{prefix}_{int(cv2.getTickCount() % 1e9)}.jpg")
    try:
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            raise IOError("图像编码失败")
        with open(save_path, 'wb') as f:
            f.write(buffer)
        print(f"Success! 结果图已保存: {save_path}")
    except Exception as e:
        print(f"Error! 结果图保存失败: {e}")


# --------------------------------------- 业务功能 ---------------------------------------
def register_face_from_image():
    """录入人脸：读图 -> 检测 -> 截取ROI -> 提特征 -> 入库"""
    image_path = input("请输入人脸图像路径: ").strip()
    image = read_image(image_path)
    if image is None:
        return

    face_id = input("请输入该人的姓名(身份标签): ").strip()
    if not face_id:
        print("Error! 姓名不能为空")
        return

    # 人脸检测 + ROI截取
    faces = face_detector.detect_faces(image)
    if not faces:
        print("未检测到人脸")
        return
    print(f"检测到 {len(faces)} 张人脸")

    # 逐张提特征并入库
    for roi, (x, y, w, h) in face_detector.crop_face_rois(image, faces):
        feature = FaceRecognition.extract_feature(roi)
        if feature is not None:
            face_db.save_feature(feature, face_id)

def recognize_from_image():
    """图片识别：读图 -> 检测 -> 截取ROI -> 逐张比对 -> 画框标注 -> 保存结果图"""
    image_path = input("请输入要识别的图像路径: ").strip()
    image = read_image(image_path)
    if image is None:
        return

    # 人脸检测 + ROI截取
    faces = face_detector.detect_faces(image)
    if not faces:
        print("未检测到人脸")
        return
    print(f"检测到 {len(faces)} 张人脸")

    # 逐张识别
    results = []
    for roi, (x, y, w, h) in face_detector.crop_face_rois(image, faces):
        results.append(FaceRecognition.recognize_face(roi))

    # 可视化 + 保存
    drawn = draw_results(image.copy(), faces, results)
    cv2.imshow("识别结果(按任意键关闭)", drawn)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    save_result(drawn, prefix='recognize')

    recognized = [r['name'] for r in results if r]
    print(f"识别完成: {len(recognized)}/{len(results)} 人匹配成功" +
          (f", 结果: {recognized}" if recognized else ""))

def recognize_from_camera():
    """摄像头实时识别: 循环 读帧 -> 检测 -> 截取 -> 比对 -> 标注 -> 刷新显示"""
    # 打开摄像头 + 合法性校验
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error! 摄像头打开失败")
        return

    print("实时识别启动, 按 'q' 退出")
    while True:
        # 循环读取视频帧
        ret, frame = cap.read()
        if not ret:
            print("Error! 无法读取摄像头帧")
            break

        # 人脸检测 + ROI截取
        faces = face_detector.detect_faces(frame)
        results = []
        for roi, (x, y, w, h) in face_detector.crop_face_rois(frame, faces):
            results.append(FaceRecognition.recognize_face(roi))   # 比对特征

        # 标注 + 刷新画面
        drawn = draw_results(frame, faces, results)
        cv2.putText(drawn, "Press 'q' to quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Real-time Face Recognition", drawn)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

def register_face_from_camera():
    """摄像头录入人脸: 按 's' 保存当前帧人脸特征入库, 按 'q' 退出"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error! 摄像头打开失败")
        return

    print("=" * 45)
    print("摄像头人脸采集: 按 's' 录入, 按 'q' 退出")
    print("=" * 45)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error! 无法读取摄像头帧")
            break
            
        faces = face_detector.detect_faces(frame)
        # 画框预览
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Press 's' to save, 'q' to quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Face Register", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if not faces:
                print("Warning! 未检测到人脸，请调整姿势")
                continue
            face_id = input("请输入人脸ID(姓名): ").strip()
            if not face_id:
                print("Warning! 人脸ID不能为空")
                continue
            for roi, (x, y, w, h) in face_detector.crop_face_rois(frame, faces):
                feature = FaceRecognition.extract_feature(roi)
                if feature is not None:
                    face_db.save_feature(feature, face_id)
        elif key == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

def show_all_features():
    """查看特征库: 共几人、每条记录信息"""
    features, info_list = face_db.get_all_features()
    if features is None:
        print("特征库为空, 请先录入人脸")
        return
    print(f"特征库共 {features.shape[0]} 条记录, 特征矩阵形状: {features.shape}")
    for info in info_list:
        print(f"  记录ID={info['id']}, 姓名={info['name']}")

def delete_face():
    """删除某人的特征(录错重录时用)"""
    face_id = input("请输入要删除的姓名: ").strip()
    if face_id:
        face_db.delete_by_name(face_id)


# --------------------------------------- 业务功能 ---------------------------------------
if __name__ == '__main__':
    while True:
        print("#" * 20 + " 人脸检测识别系统" + "#" * 20)
        print("1. 通过图片录入人脸")
        print("2. 通过摄像头录入人脸")
        print("3. 查看特征库")
        print("4. 从图片识别人脸")
        print("5. 摄像头实时识别人脸")
        print("6. 删除已录入人脸")
        print("0. 退出系统")
        choice = input("请输入选项: ").strip()

        if choice == "1":
            register_face_from_image()
        elif choice == "2":
            register_face_from_camera()
        elif choice == "3":
            show_all_features()
        elif choice == "4":
            recognize_from_image()
        elif choice == "5":
            recognize_from_camera()
        elif choice == "6":
            delete_face()
        elif choice == "0":
            face_db.close()
            print("系统已退出")
            break
        else:
            print("Error! 选项不存在, 请重新输入")
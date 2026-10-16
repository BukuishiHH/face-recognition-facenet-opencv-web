"""
Step 2-4 CLI 业务流程模块单元测试
被测模块: backend/main.py
说明: 摄像头相关函数(register_face_from_camera / recognize_from_camera)依赖物理
      摄像头与 GUI 窗口, 按 temp/TEST_SAMPLES_GUIDE.md 约定不纳入自动化测试。
"""

import builtins
import runpy

import cv2
import numpy as np
import pytest

import main

FEATURE_DIM = 512


# --------------------------------------- read_image ---------------------------------------
class TestReadImage:

    def test_read_valid_image(self, one_face_path):
        img = main.read_image(one_face_path)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3 and img.shape[2] == 3

    def test_missing_file_returns_none(self, tmp_path):
        assert main.read_image(str(tmp_path / 'not_exist.jpg')) is None

    def test_corrupt_file_returns_none(self, tmp_path):
        bad = tmp_path / 'corrupt.jpg'
        bad.write_bytes(b'this is not an image')
        assert main.read_image(str(bad)) is None

    def test_chinese_path_supported(self, tmp_path, one_face_path):
        """中文路径读取: np.fromfile + cv2.imdecode 方案"""
        dst = tmp_path / '中文目录' / '人脸样例.jpg'
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(one_face_path, 'rb') as src:
            data = src.read()
        with open(dst, 'wb') as f:
            f.write(data)
        assert main.read_image(str(dst)) is not None


# --------------------------------------- put_chinese_text ---------------------------------------
class TestPutChineseText:

    def test_shape_preserved(self):
        img = np.zeros((80, 240, 3), np.uint8)
        out = main.put_chinese_text(img, '人脸检测', (5, 5))
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_pixels_are_drawn(self):
        img = np.zeros((80, 240, 3), np.uint8)
        out = main.put_chinese_text(img, '识别成功', (5, 5), font_size=22, color=(0, 255, 0))
        assert not np.array_equal(img, out)

    def test_bgr_color_mapping(self):
        """颜色修复校验: 传入 BGR 红色 (0,0,255), 渲染像素 R 通道应大于 B 通道"""
        img = np.zeros((80, 240, 3), np.uint8)
        out = main.put_chinese_text(img, '红', (5, 5), font_size=30, color=(0, 0, 255))
        mask = out.sum(axis=2) > 0
        assert mask.any(), "未绘制任何像素"
        pixels = out[mask]                                 # BGR 排列
        assert pixels[:, 2].mean() > pixels[:, 0].mean()


# --------------------------------------- draw_results ---------------------------------------
class TestDrawResults:

    def test_green_box_for_known_face(self):
        img = np.zeros((300, 300, 3), np.uint8)
        faces = [(10, 10, 60, 60)]
        results = [{'name': '张三', 'similarity': 0.92}]
        out = main.draw_results(img.copy(), faces, results)
        assert tuple(out[10, 10]) == (0, 255, 0)           # 识别成功 -> 绿框

    def test_red_box_for_unknown_face(self):
        img = np.zeros((300, 300, 3), np.uint8)
        faces = [(10, 10, 60, 60)]
        out = main.draw_results(img.copy(), faces, [None])
        assert tuple(out[10, 10]) == (0, 0, 255)           # 无匹配 -> 红框

    def test_multiple_faces_drawn(self):
        img = np.zeros((400, 400, 3), np.uint8)
        faces = [(10, 10, 50, 50), (200, 200, 50, 50)]
        results = [{'name': 'A', 'similarity': 0.9}, None]
        out = main.draw_results(img.copy(), faces, results)
        assert tuple(out[10, 10]) == (0, 255, 0)
        assert tuple(out[200, 200]) == (0, 0, 255)


# --------------------------------------- save_result ---------------------------------------
class TestSaveResult:

    def test_save_creates_readable_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main, 'RESULT_DIR', str(tmp_path))
        img = np.zeros((60, 60, 3), np.uint8)
        main.save_result(img, prefix='unit_test')

        saved = list(tmp_path.glob('unit_test_*.jpg'))
        assert len(saved) == 1
        reloaded = main.read_image(str(saved[0]))
        assert reloaded is not None and reloaded.shape == img.shape


# --------------------------------------- 特征库菜单功能 ---------------------------------------
class TestLibraryMenu:

    def test_show_all_features_empty(self, isolated_db, capsys):
        main.show_all_features()
        assert '特征库为空' in capsys.readouterr().out

    def test_show_all_features_with_records(self, isolated_db, capsys):
        isolated_db.save_feature(np.random.rand(1, FEATURE_DIM).astype(np.float32), '张三')
        main.show_all_features()
        out = capsys.readouterr().out
        assert '共 1 条记录' in out
        assert '张三' in out

    def test_delete_face(self, isolated_db, monkeypatch):
        isolated_db.save_feature(np.random.rand(1, FEATURE_DIM).astype(np.float32), '待删除')
        monkeypatch.setattr(builtins, 'input', lambda prompt='': '待删除')
        main.delete_face()
        matrix, _ = isolated_db.get_all_features()
        assert matrix is None                              # 记录已删除


# --------------------------------------- 业务编排 ---------------------------------------
class TestBusinessFlow:

    def test_register_face_from_image(self, isolated_db, monkeypatch, one_face_path, capsys):
        answers = iter([one_face_path, '测试员'])
        monkeypatch.setattr(builtins, 'input', lambda prompt='': next(answers))
        main.register_face_from_image()

        matrix, info = isolated_db.get_all_features()
        assert matrix is not None
        assert info[0]['name'] == '测试员'
        assert '已入库' in capsys.readouterr().out

    def test_register_aborts_on_empty_name(self, isolated_db, monkeypatch, one_face_path, capsys):
        answers = iter([one_face_path, ''])
        monkeypatch.setattr(builtins, 'input', lambda prompt='': next(answers))
        main.register_face_from_image()
        assert isolated_db.get_all_features() == (None, None)
        assert '姓名不能为空' in capsys.readouterr().out

    def test_register_aborts_on_missing_file(self, isolated_db, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(builtins, 'input', lambda prompt='': str(tmp_path / 'none.jpg'))
        main.register_face_from_image()
        assert '图片文件不存在' in capsys.readouterr().out

    def test_recognize_from_image(self, isolated_db, monkeypatch, one_face_path, tmp_path, capsys):
        # 1. 先录入同一个人的人脸特征
        from face_recognition import FaceRecognition
        img = main.read_image(one_face_path)
        faces = main.face_detector.detect_faces(img)
        roi = main.face_detector.crop_face_rois(img, faces)[0][0]
        isolated_db.save_feature(FaceRecognition.extract_feature(roi), '测试员')

        # 2. 关闭 GUI 显示, 结果图落到临时目录
        monkeypatch.setattr(main, 'RESULT_DIR', str(tmp_path))
        monkeypatch.setattr(cv2, 'imshow', lambda *a, **k: None)
        monkeypatch.setattr(cv2, 'waitKey', lambda *a, **k: ord('q'))
        monkeypatch.setattr(cv2, 'destroyAllWindows', lambda *a, **k: None)
        monkeypatch.setattr(builtins, 'input', lambda prompt='': one_face_path)

        main.recognize_from_image()

        out = capsys.readouterr().out
        assert '识别完成' in out
        assert '测试员' in out
        assert list(tmp_path.glob('recognize_*.jpg'))      # 结果图已保存


# --------------------------------------- 主菜单 ---------------------------------------
class TestMainMenu:

    def test_invalid_choice_then_exit(self, isolated_db, monkeypatch, capsys):
        """主菜单: 非法选项给出提示, 输入 0 正常退出"""
        answers = iter(['9', '0'])                         # 先非法选项, 再退出
        monkeypatch.setattr(builtins, 'input', lambda prompt='': next(answers))
        runpy.run_path(main.__file__, run_name='__main__')

        out = capsys.readouterr().out
        assert '人脸检测识别系统' in out
        assert '选项不存在' in out
        assert '系统已退出' in out

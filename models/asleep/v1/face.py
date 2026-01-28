import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Iterable, Union, Tuple, List, Optional

# 设置环境变量以抑制 TensorFlow/absl 的冗长日志（存在时有效）
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')
os.environ['MEDIAPIPE_DISABLE_XNNPACK'] = '1'

try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass

# mediapipe 是可选依赖：如果不存在，模块仍能被导入，但 faceCheck_getEAR 会退化为返回未检测到
try:
    import mediapipe as mp
    mp_facemesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    # 这个内部方法有时不存在，使用保护性获取
    print("✅ mediapipe 导入成功！mp_facemesh：", mp_facemesh)
    denormalize_coordinates = getattr(mp_drawing, '_normalized_to_pixel_coordinates', None)
except Exception as e:
    print(f"❌ mediapipe 导入失败（原因：{e}）")
    mp = None
    mp_facemesh = None
    mp_drawing = None
    denormalize_coordinates = None


def distance(point_1, point_2):
    """计算两点之间的欧氏距离（L2范数）"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


# get_ear (…)函数将.landmark属性作为参数。在每个索引位置，我们都有一个NormalizedLandmark对象。该对象保存标准化的x、y和z坐标值。
def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    """
    计算单眼的眼睛纵横比（Eye Aspect Ratio）
    
    参数:
        landmarks: (列表) 检测到的关键点列表
        refer_idxs: (列表) 按顺序排列的关键点索引位置 P1, P2, P3, P4, P5, P6
        frame_width: (整数) 捕获帧的宽度
        frame_height: (整数) 捕获帧的高度
    
    返回:
        ear: (浮点数) 眼睛纵横比
    """
    try:
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, 
                                             frame_width, frame_height)
            coords_points.append(coord)

        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points

def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    """计算左右眼的平均眼睛纵横比"""

    left_ear, left_lm_coordinates = get_ear(
                                      landmarks, 
                                      left_eye_idxs, 
                                      image_w, 
                                      image_h
                                    )
    right_ear, right_lm_coordinates = get_ear(
                                      landmarks, 
                                      right_eye_idxs, 
                                      image_w, 
                                      image_h
                                    )
    Avg_EAR = (left_ear + right_ear) / 2.0

    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)

def video_test():
    """视频测试函数，处理视频中的每一帧并输出为新的MP4视频"""
    # 读取视频文件
    cap = cv2.VideoCapture("/data/cxc/code/车联网/data/睡觉面部测试.mp4")
    
    # 获取原视频的属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 设置输出视频的编码器和参数
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码器
    out_path = "/data/cxc/code/车联网/data/output_video.mp4"
    out = cv2.VideoWriter(out_path, fourcc, fps, (640, 480))
    
    # 设置眼睛关键点索引
    chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
    chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
    
    # 初始化人脸网格检测器（视频模式）
    with mp_facemesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 转换颜色空间（OpenCV使用BGR，MediaPipe使用RGB）
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 调整大小
            rgb_frame = cv2.resize(rgb_frame, (640, 480))
            
            # 处理图像
            results = face_mesh.process(rgb_frame)
            
            # 将RGB帧转回BGR用于OpenCV显示
            processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # 如果检测到人脸
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 获取图像尺寸
                    h, w, _ = processed_frame.shape
                    
                    # 计算EAR值
                    EAR, _ = calculate_avg_ear(
                        face_landmarks.landmark,
                        chosen_left_eye_idxs,
                        chosen_right_eye_idxs,
                        w, h
                    )
                    
                    # 在图像上显示EAR值
                    cv2.putText(processed_frame,
                                f"EAR: {round(EAR, 2)}", (10, 30),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 255, 0), 2)
                    
                    # 根据EAR值判断眼睛状态
                    eye_state = "闭眼" if EAR < 0.2 else "睁眼"
                    cv2.putText(processed_frame,
                                f"状态: {eye_state}", (10, 60),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 255, 0), 2)
            
            # 将处理后的帧写入输出视频
            out.write(processed_frame)
            
            # 显示实时处理结果（可选）
            # cv2.imshow('Eye State Detection', processed_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"视频处理完成，已保存到：{out_path}")


def _process_image_list(images: List[np.ndarray], *, static_image_mode: bool = True,
                        min_detection_confidence: float = 0.5,
                        min_tracking_confidence: float = 0.5,
                        chosen_left_eye_idxs=None,
                        chosen_right_eye_idxs=None) -> Tuple[int, Optional[List[int]]]:
    """Process a list of images (numpy arrays). Returns (flag, bbox).

    flag: 1 if fatigue detected (EAR threshold count), else 0.
    bbox: last detected face bbox or None.
    """
    if chosen_left_eye_idxs is None:
        chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
    if chosen_right_eye_idxs is None:
        chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]

    if mp_facemesh is None:
        print("Warning: mediapipe not installed; faceCheck_getEAR will return (0, None)")
        return 0, None

    count = 0
    face_bbox = None

    with mp_facemesh.FaceMesh(static_image_mode=static_image_mode,
                              refine_landmarks=True,
                              max_num_faces=1,
                              min_detection_confidence=min_detection_confidence,
                              min_tracking_confidence=min_tracking_confidence) as face_mesh:
        for image in images:
            if image is None:
                continue
            image = np.ascontiguousarray(image)
            imgH, imgW = image.shape[:2]

            if image.shape[2] == 3:
                proc_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                proc_img = image

            results = face_mesh.process(proc_img)
            multi = getattr(results, 'multi_face_landmarks', None)
            if not multi:
                continue

            for face_landmarks in multi:
                landmarks = face_landmarks.landmark
                EAR, lm_coords = calculate_avg_ear(landmarks,
                                                   chosen_left_eye_idxs,
                                                   chosen_right_eye_idxs,
                                                   imgW, imgH)
                if EAR < 0.2:
                    count += 1

                xs = []
                ys = []
                for lm in landmarks:
                    if denormalize_coordinates is None:
                        continue
                    coord = denormalize_coordinates(lm.x, lm.y, imgW, imgH)
                    if coord:
                        x, y = coord
                        xs.append(x)
                        ys.append(y)
                if xs and ys:
                    x1 = max(0, min(xs))
                    y1 = max(0, min(ys))
                    x2 = min(imgW, max(xs))
                    y2 = min(imgH, max(ys))
                    face_bbox = [int(x1), int(y1), int(x2), int(y2)]

    return (1, face_bbox) if count > 3 else (0, face_bbox)


def process_video(video_path: str, *, max_frames: Optional[int] = None,
                  chosen_left_eye_idxs=None, chosen_right_eye_idxs=None) -> Tuple[int, Optional[List[int]]]:
    """Process a video file path and return (flag, bbox)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: cannot open video: {video_path}")
        return 0, None

    def frame_generator():
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
            i += 1
            if max_frames and i >= max_frames:
                break
        cap.release()

    return _process_frames_generator(frame_generator(), static_image_mode=False,
                                      chosen_left_eye_idxs=chosen_left_eye_idxs,
                                      chosen_right_eye_idxs=chosen_right_eye_idxs)


def _process_frames_generator(frames: Iterable[np.ndarray], *, static_image_mode: bool = False,
                               chosen_left_eye_idxs=None, chosen_right_eye_idxs=None,
                               min_detection_confidence: float = 0.5,
                               min_tracking_confidence: float = 0.5) -> Tuple[int, Optional[List[int]]]:
    """Process frames from an iterable/generator. Returns (flag, bbox)."""

    batch = []
    BATCH_SIZE = 16
    for frame in frames:
        batch.append(frame)
        if len(batch) >= BATCH_SIZE:
            flag, bbox = _process_image_list(batch, static_image_mode=static_image_mode,
                                             min_detection_confidence=min_detection_confidence,
                                             min_tracking_confidence=min_tracking_confidence,
                                             chosen_left_eye_idxs=chosen_left_eye_idxs,
                                             chosen_right_eye_idxs=chosen_right_eye_idxs)

            if flag == 1:
                return flag, bbox
            batch = []
    if batch:
        return _process_image_list(batch, static_image_mode=static_image_mode,
                                   min_detection_confidence=min_detection_confidence,
                                   min_tracking_confidence=min_tracking_confidence,
                                   chosen_left_eye_idxs=chosen_left_eye_idxs,
                                   chosen_right_eye_idxs=chosen_right_eye_idxs)

    return 0, None


def faceCheck_getEAR(input_data: Union[str, np.ndarray, Iterable[np.ndarray]]) -> Tuple[int, Optional[List[int]]]:
    """Flexible entry point.

    Accepts:
      - video file path (str)
      - single image (`np.ndarray`)
      - list of images (`Iterable` of `np.ndarray`)
      - frame generator/iterator

    Returns (flag, bbox) like before.
    """

    if isinstance(input_data, str):
        return process_video(input_data)

    if isinstance(input_data, np.ndarray):
        return _process_image_list([input_data])


    if isinstance(input_data, Iterable):

        try:
            if not hasattr(input_data, '__next__'):
                images = list(input_data)
                return _process_image_list(images)
            else:
                return _process_frames_generator(input_data)
        except Exception:
            return 0, None

    return 0, None


if __name__ == "__main__":
    # get_eye_landmarks()
    # test()
    pass
    # video_test()  
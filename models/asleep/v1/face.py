import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# 设置环境变量以抑制 TensorFlow/absl 的冗长日志（存在时有效）
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')
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
    denormalize_coordinates = getattr(mp_drawing, '_normalized_to_pixel_coordinates', None)
except Exception:
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
        # Compute the euclidean distance between the horizontal
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, 
                                             frame_width, frame_height)
            coords_points.append(coord)

        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Compute the eye aspect ratio
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


def faceCheck_getEAR(images_path):
    # print(type(images_path))
    """测试函数，用于验证EAR算法在眼睛打开和闭合状态下的效果"""

    chosen_left_eye_idxs  = [362, 385, 387, 263, 373, 380]  # 左眼周围的6个特定关键点索引
    chosen_right_eye_idxs = [33,  160, 158, 133, 153, 144]  # 右眼周围的6个特定关键点索引
    # 加载测试图像
    # image_eyes_open  = cv2.imread("/data/cxc/code/车联网/data/睁眼.jpg")    # 眼睛打开的图像
    # image_eyes_close = cv2.imread("/data/cxc/code/车联网/data/自己闭眼.jpg")  # 眼睛闭合的图像


    # 如果 mediapipe 不可用，直接返回未检测到疲劳
    if mp_facemesh is None:
        print("Warning: mediapipe not installed; faceCheck_getEAR will return (0, None)")
        return 0, None

    try:
        # 遍历测试图像
        count  = 0
        for idx, image in enumerate(images_path):

            # 确保图像数据连续存储在内存中，提高处理效率
            image = np.ascontiguousarray(image)
            imgH, imgW, _ = image.shape

            # 创建原始图像的副本，用于绘制EAR值
            # custom_chosen_lmk_image = image.copy()

            # 使用static_image_mode运行人脸网格检测
            with mp_facemesh.FaceMesh(refine_landmarks=True) as face_mesh:
                results = face_mesh.process(image).multi_face_landmarks

                # 如果检测到人脸
                if results:
                            for face_id, face_landmarks in enumerate(results):
                                landmarks = face_landmarks.landmark
                                # 计算平均眼睛纵横比
                                EAR, lm_coords = calculate_avg_ear(
                                        landmarks,
                                        chosen_left_eye_idxs,
                                        chosen_right_eye_idxs,
                                        imgW,
                                        imgH
                                    )

                                if EAR < 0.2:
                                    count += 1

                                # 计算人脸 bbox（使用所有关键点的最小外接矩形）
                                xs = []
                                ys = []
                                for lm in landmarks:
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
                                else:
                                    face_bbox = None
                        # 在图像上绘制EAR值
                        # cv2.putText(custom_chosen_lmk_image, 
                        #             f"EAR: {round(EAR, 2)}", (1, 24),
                        #             cv2.FONT_HERSHEY_COMPLEX, 
                        #             0.9, (255, 255, 255), 2
                        # )                
                        # 保存结果图像
                        # cv2.imwrite('/data/cxc/code/车联网/data/' + str(idx) + '.png', custom_chosen_lmk_image)
                        # plot(img_dt=image.copy(),
                        #     img_eye_lmks_chosen=custom_chosen_lmk_image,
                        #     face_landmarks=face_landmarks,
                        #     ts_thickness=1, 
                        #     ts_circle_radius=3, 
                        #     lmk_circle_radius=3
                        # )
        if count > 3:
            return 1, face_bbox
        else:
            return 0, face_bbox
    except Exception as e:
        print(e)



if __name__ == "__main__":
    # get_eye_landmarks()
    # test()
    pass
    # video_test()  
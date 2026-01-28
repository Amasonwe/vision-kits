from models.asleep.v1.face import faceCheck_getEAR
# 替换为你的视频路径
flag, bbox = faceCheck_getEAR(r"tmp\3b9d122218194f5cbffca19835a06958.mp4")
print(f"检测结果：flag={flag}, bbox={bbox}")
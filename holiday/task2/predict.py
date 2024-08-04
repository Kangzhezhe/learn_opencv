# 导入必要的库
import cv2
from ultralytics import YOLO

# 加载 YOLO 模型
model = YOLO('/home/kangzhehao/Documents/holiday/task2/runs/detect/train/weights/best.pt')

# 设置输入视频的路径
video_path = "./深度学习任务二测试视频.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频的帧宽度、帧高度和帧速率
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# 定义编解码器和输出视频文件
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./output1.mp4', fourcc, fps, (frame_width, frame_height))

# 循环遍历视频帧
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # 使用 YOLO 模型进行目标检测或跟踪
        results = model.track(frame, persist=True)
        # 绘制带注释的帧
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
    else:
        break

# 释放视频捕获和视频写入对象
cap.release()
out.release()
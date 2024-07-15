import cv2
from ultralytics import YOLO
from tqdm import tqdm   # 进度条

# 加载模型
#model = YOLO("C:\\Users\\liruilong\\.yolo_model\\yolov8x-pose-p6.pt")
model = YOLO("C:\\Users\\liruilong\\.yolo_model\\yolov8x.pt")
# 打开视频文件
 
video_path ="c:\\Users\\Administrator\\Videos\\20240716.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频帧的维度
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output1RR.mp4', fourcc, 25.0, (frame_width, frame_height))

# 设置整个视频处理的进度条 
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pbar = tqdm(total=total_frames, desc="Processing video", unit="frames")

# 处理视频帧
for _ in range(total_frames):
    # 读取某一帧
    success, frame = cap.read()
    if success:
        # 使用yolov8进行预测
        results = model(frame)
        # 可视化结果
        annotated_frame = results[0].plot(boxes=False,labels=True,line_width=1)
        #annotated_frame = results[0].plot(kpt_radius=5,labels=False,boxes=False,line_width=1)
        # 将带注释的帧写入视频文件
        out.write(annotated_frame)
        # 更新进度条
        pbar.update(1)
    else:
        # 最后结尾中断视频帧循环
        break

# 若有部分帧未正常打开，进度条是不会达到百分之百的，下面这行代码会让进度条跑满
pbar.update(total_frames - pbar.n)
# 完成视频处理，关闭进度条
pbar.close()

# 释放读取和写入对象
cap.release()
out.release()


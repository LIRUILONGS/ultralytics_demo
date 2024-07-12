from ultralytics import YOLO
from PIL import Image
import cv2
import os

from imutils import paths
 
model = YOLO("C:\\Users\\liruilong\\.yolo_model\\yolov8x-pose-p6.pt")

#model = YOLO("C:\\Users\\liruilong\\.yolo_model\\yolov8s.pt")
# from ndarray
results = []
print("开始处理照片")

result = model("c:\\Users\\liruilong\\Documents\\Adobe\\Premiere Pro\\23.0\\test.mp4") 

#cv2.imshow("result", results)
#cv2.imwrite( "frame.jpg", results)

#for result_ in  results:
#    result,fn = result_


        
for r in result:
    im_array = r.plot(kpt_radius=5,labels=False,boxes=False,line_width=1)  # plot a BGR numpy array of predictions
    #im_array = r.plot(kpt_radius=15,line_width=1,labels=False,boxes=False,)  # plot a BGR numpy array of predictions
    #im_array = r.plot(line_width=2) 
    print(r)
    im_array = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    #im_array.show()  # show image
    #im_array.save("C:\\Users\\liruilong\\Documents\\GitHub\\ultralytics_demo\\examples\\demo\\images\\346\\27"+fn)
    #im_array.save("./images/"+ip+"/"+ip+"_yp.jpg")  # save image

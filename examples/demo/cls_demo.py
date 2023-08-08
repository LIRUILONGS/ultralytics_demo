from ultralytics import YOLO
from PIL import Image
import cv2
import os

from imutils import paths
 
#model = YOLO("C:\\Users\\liruilong\\.yolo_model\\yolov8x-pose-p6.pt")

model = YOLO("C:\\Users\\liruilong\\.yolo_model\\yolov8s.pt")
# from ndarray
results = []
for fn in paths.list_images("./185"):
    im2 = cv2.imread(fn)

#results = model(im2)  # predict on an image
    result = model(im2) 
    print(fn)  
    results.append((result,fn))
#cv2.imshow("result", results)
#cv2.imwrite( "frame.jpg", results)

for result_ in  results:
    result,fn = result_
    fn = os.path.basename(fn)
    ip = str(fn).split("_01_")[0]
        
    for r in result:
        #im_array = r.plot(kpt_radius=3,labels=False,boxes=False,line_width=1)  # plot a BGR numpy array of predictions
        #im_array = r.plot(line_width=3,labels=False,boxes=False,)  # plot a BGR numpy array of predictions
        im_array = r.plot(line_width=3) 
        im_array = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        #im_array.show()  # show image
        im_array.save("./images/"+ip+"/"+ip+"_yc.jpg")  # save image

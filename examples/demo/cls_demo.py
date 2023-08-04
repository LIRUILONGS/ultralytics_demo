from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("C:\\Users\\liruilong\\.yolo_model\\yolov8x-pose-p6.pt")


# from ndarray
im2 = cv2.imread("Y:\\image8.jpg")

#results = model(im2)  # predict on an image
results = model(im2)

#cv2.imshow("result", results)
#cv2.imwrite( "frame.jpg", results)

for r in results:
    #im_array = r.plot(kpt_radius=3,labels=False,boxes=False,line_width=1)  # plot a BGR numpy array of predictions
    im_array = r.plot(line_width=2)  # plot a BGR numpy array of predictions
    im_array = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im_array.show()  # show image
    im_array.save('results.jpg')  # save image

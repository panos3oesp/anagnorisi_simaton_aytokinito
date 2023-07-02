#pip install --upgrade torch==1.9.0
#pip install --upgrade torchvision==0.10.0
#pip install --upgrade torchaudio==0.9.0
#git clone https://github.com/ultralytics/yolov5
#cd yolov5
#pip install -r requirements.txt
from mplayer import Player
import torch
import numpy as np
import time
import cv2
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression



player = Player()
# Load the YOLOv5 model
model_path = "./weights/best-small.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load(model_path, device)

# Set the model to evaluation mode
model.eval()

print(model.names)
# print(dir(model))

# Initialize the video capture object
cap = cv2.VideoCapture(0)
success, img = cap.read()

#cap = cv2.VideoCapture(2)

cap.set(cv2.CAP_PROP_FPS, 1)
fps = int(cap.get(12))
print("fps:", fps)



import cv2
#cap=cv2.VideoCapture(-1)
cap.set(3,640)#width
cap.set(4,480)#height
cap.set(10,100)#brightness
'''
while True:
    success, img = cap.read()
    if success:
        print(img)#getting None
        print(success)#getting False
        cv2.imshow("video", img)
        cv2.waitKey(1)
        if 0xFF == ord('q') :
            break

'''

# Set the resolution to 320x240
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
doIt = True
prev = ''
while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()
    if True:
        cap.release()
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.set(cv2.CAP_PROP_FPS, 1)
        fps = int(cap.get(12))
        print("fps:", fps)
        cap.set(3,640)#width
        cap.set(4,480)#height
        cap.set(10,100)#brightness
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            
    if doIt:
        
    
        print('start')
        # Resize the image to the input size of the model
        frame_size = 480
        frame_height = 480
        frame = cv2.resize(frame, (frame_size, frame_height))
        
        # Convert the frame to a PyTorch tensor
        frame = torch.from_numpy(frame.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)

        # Make the prediction
        with torch.no_grad():
            outputs = model(frame.to(device))

        # Apply non-maximum suppression to get the final detections
        conf_thresh = 0.35
        iou_thresh = 0.5
        detections = non_max_suppression(outputs, conf_thresh, iou_thresh)
        

        # Convert the PyTorch tensor back to a NumPy array
        frame = frame.squeeze(0).permute(1, 2, 0).numpy()

        # Draw bounding boxes on the image with the predicted class and percentage confidence for each detection
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (0, 0, 255)
        
        
        selDetection=None
        doIt = False
        maxConf = 0
        for detection in detections[0]:
            x1, y1, x2, y2, conf, cls = detection.tolist()
            if conf > maxConf:
                selDetection = detection
        if len(detections[0]):
            if selDetection == None:
                x1, y1, x2, y2, conf, cls = detections[0][0].tolist()
            else:
                x1, y1, x2, y2, conf, cls = selDetection.tolist()    
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls_name = model.names[int(cls)]
            print(cls_name)
            
            
            if cls_name=="stop":                
                soundPath = 'sounds/Stop.m4a'
            elif cls_name=="bike":                
                soundPath = 'sounds/Bike.m4a'
            elif cls_name=="speed_limit_30":                
                soundPath = 'sounds/Limit30.m4a'
            elif cls_name=="speed_limit_50":                
                soundPath = 'sounds/Limit50.m4a'
            print(conf)
            if cls_name != prev and conf>0.69:                
                print(soundPath)                
                player.loadfile(soundPath)
                prev = cls_name
                time.sleep(5)
                cap.release()
            doIt = True
        if len(detections[0])==0:
            doIt = True
            
        #label = f"{cls_name} {conf:.2f}"
        #(w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        #cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        #cv2.rectangle(frame, (x1, y1 - 20 - h), (x1 + w, y1 - 20), color, cv2.FILLED)
        #cv2.putText(frame, label, (x1, y1 - 10), font, font_scale, (0, 0, 0), thickness)
    
    # Display the frame
    #cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
    

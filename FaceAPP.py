import dlib
from PIL import Image, ImageOps
import face_recognition
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import os

knownfaces_dir = "known_faces"
TOLERANCE = 0.30
FRAME_THICKNESS = 3
FONT_THICKNESS=2
MODEL1 = "cnn"
MODEL2 = "hog"

def crop_image(im,face_location):
    img = Image.fromarray(im, 'RGB')
    img_cropped = img.crop((face_location[0][3]-50, face_location[0][0]-50, face_location[0][1]+50, face_location[0][2]+50)) 
    image=np.asarray(img_cropped)
    return image

print("loading known faces")
known_faces = []
knwon_names = []
for name in os.listdir(knownfaces_dir):
    for filename in os.listdir(knownfaces_dir+"/"+name): #join
        image = face_recognition.load_image_file(knownfaces_dir+"/"+name+"/"+filename)
        print(knownfaces_dir+"/"+name+"/"+filename)
        
        #image resize
        #image=Image.fromarray(image)
        #image=image.resize((300,250))
        #image=np.asarray(image)
        #plt.imshow(image)
        #plt.show()
        #image encoding
        encoding = face_recognition.face_encodings(crop_image(image,locations))
        known_faces.append(encoding[0])
        knwon_names.append(name)

def read_camera(image):
    #Image resizing
    image=Image.fromarray(image)
    image=image.resize((300,250))
    image=np.asarray(image)
    locations = face_recognition.face_locations(image,model=MODEL2)
    if len(locations)>0:
        encodings = face_recognition.face_encodings(image,locations)

        for face_encodings,face_location in zip(encodings, locations):
            results =face_recognition.compare_faces(known_faces,face_encodings,TOLERANCE)
            match = None
            if True in results:
                match= knwon_names[results.index(True)]
                print(f"Found: {match}")
                top_left = (face_location[3],face_location[0])
                bottom_right = (face_location[1],face_location[2])
                color = [0,255,0]
                cv2.rectangle(image,top_left,bottom_right,color,FRAME_THICKNESS)

                top_left = (face_location[3],face_location[2])
                bottom_right = (face_location[1],face_location[2]+22)
                cv2.rectangle(image,top_left,bottom_right,color,FRAME_THICKNESS)
                cv2.putText(image,match,(face_location[3]+10,face_location[2]+15),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),FONT_THICKNESS)
    cv2.imshow("output",image)
    cv2.waitKey(1)

import cv2 as cv
import numpy as np
from PIL import Image

cap = cv.VideoCapture(0)
# Variable for color to draw optical flow track
color = (0, 0, 255)
i=1
while(cap.isOpened()):
    ret, first_frame = cap.read()
    read_camera(first_frame)
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()
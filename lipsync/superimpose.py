import cv2
import numpy as np
import mediapipe as mp
import os
import ast
from ruamel.yaml import YAML

mp_drawing = mp.solutions.drawing_utils

detector = mp.solutions.face_mesh.FaceMesh(
max_num_faces=10,
refine_landmarks=True,
min_detection_confidence=0.5,
min_tracking_confidence=0.5
)

mp_face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5)

arr_mouthPoints = []
arr_timepoints =[]
frame_no = 10
arr_mouth = []
frame_counter = 1

outer_pts = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
inner_pts = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

def superimpose():
    cap1 = cv2.VideoCapture("/home/chathushkavi/Downloads/superimpose/scene1.mp4")
    cap2 = cv2.VideoCapture("/home/chathushkavi/Downloads/superimpose/maheshbau_telugu.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # cap.get(3) is the video width, cap.get(4) is the video height.
    cap_width = int(cap2.get(3))
    cap_height = int(cap2.get(4))

    fps = cap2.get(cv2.CAP_PROP_FPS)

    approx_frame_count = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    approx_frame_count_x = approx_frame_count - 1
    print("Total frames: {}".format(approx_frame_count))
    print("FPS :",fps)
    frame_counter = 1
    baseTime = 0.1  
    running = True

    out = cv2.VideoWriter("/home/chathushkavi/Downloads/superimpose/output_video.mp4", fourcc, fps, (cap_width, cap_height))

    while running:
        okay1  , frame1 = cap1.read()
        okay2 , frame2 = cap2.read()

        # if okay1:
        if frame1 is not None:
            frame_result = process_frame(frame1)
            keypoints = frame_result.multi_face_landmarks

            if(keypoints is not None):
                ls_single_face=keypoints[0].landmark 

                mouthPoints = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

                for i in range(len(mouthPoints)):
                    test = [(ls_single_face[mouthPoints[i]].x), (ls_single_face[mouthPoints[i]].y)]
                    arr_mouthPoints.append(test)

                points_str = str(arr_mouthPoints)

                arr_format = [round(baseTime, 1), points_str]
                arr_timepoints.append(arr_format)
                arr_mouthPoints.clear()
                res_string = str(arr_timepoints)[1:-1]
                res_list = ast.literal_eval(res_string)

                arr_timepoints.clear()

                baseTime = baseTime + 0.1

                height, width, _ = frame1.shape

                drawlips(frame2,outer_pts,ls_single_face,height, width)
                drawlips(frame2,inner_pts,ls_single_face,height, width)

            out.write(frame2)


        frame_counter = frame_counter + 1   
        if(frame_counter > approx_frame_count_x):
        # if(frame_counter > frame_no):  
            running = False
            out.release()

def process_frame(frame):
    try:
        results = detector.process(frame)
        return results
        
    except Exception as e:
        print(e)

def drawlips(frame2,pts,ls_single_face,height, width):
    for i in pts:
        pt1 = ls_single_face[i]
        x = int(pt1.x * width)
        y = int(pt1.y * height)

        coordinates = [x, y]
        arr_mouth.append(coordinates)
        
        cv2.circle(frame2, (x, y), 1, (255,0,255), -1)
    
    # new points for polygon
    # create and reshape array
    arr_mouth1 = np.array(arr_mouth)
    arr_mouth1 = arr_mouth1.reshape((-1, 1, 2))

    # Attributes
    isClosed = True
    color = (255,0,255)
    thickness = 2

    # draw closed polyline
    cv2.polylines(frame2, [arr_mouth1], isClosed, color, thickness)

    arr_mouth.clear()


if __name__ == '__main__':
    superimpose()

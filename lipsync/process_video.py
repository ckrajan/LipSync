import time
import cv2
import numpy as np
import mediapipe as mp
import json
import ast

cap = None

detector = mp.solutions.face_mesh.FaceMesh(
max_num_faces=1,
refine_landmarks=False,
min_detection_confidence=0.5,
min_tracking_confidence=0.5
)

mp_holistic = mp.solutions.holistic

arr_mouthPoints = []
arr_timepoints =[]

frame_no = 10

def run():
    """Start loop in thread capturing incoming frames.
    """
    cap = cv2.VideoCapture("/home/chathushkavi/Downloads/videoplayback.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # cap.get(3) is the video width, cap.get(4) is the video height.
    cap_width = int(cap.get(3))
    cap_height = int(cap.get(4))

    fps = cap.get(cv2.CAP_PROP_FPS)

    # out = cv2.VideoWriter("/home/chathushkavi/Downloads/output.mp4", fourcc, fps, (cap_width, cap_height))

    approx_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames: {}".format(approx_frame_count))
    print("FPS :",fps)
    running = True
    frame_counter = 0
    baseTime = 0.1

    while running:
        ret, frame = cap.read()
        # print(frame)
        frame_result = process_frame(frame)
        keypoints = frame_result.multi_face_landmarks

        if(keypoints != None):
            ls_single_face=keypoints[0].landmark
        # x_list = []
        # y_list = []
        # for id,idx in enumerate(ls_single_face):
        #     x_list.append(idx.x)
        #     y_list.append(idx.y)
            # print(id, idx.x,idx.y,idx.z)           

            mouthPoints = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

            for i in range(len(mouthPoints)):
                test = [(ls_single_face[mouthPoints[i]].x) * 436.098602295, (ls_single_face[mouthPoints[i]].y) * 563.193893433]
                arr_mouthPoints.append(test)

            points_str = str(arr_mouthPoints)

            arr_format = [round(baseTime, 1), points_str]
            arr_timepoints.append(arr_format)
            arr_mouthPoints.clear()
            res_string = str(arr_timepoints)[1:-1]
            res_list = ast.literal_eval(res_string)
            # print(res_list)

            with open('/home/chathushkavi/projects/lipsync/lipsync/target.json', 'a') as f:
                if frame_counter == 0:
                    f.write('[')
                json.dump(res_list, f)
                if frame_counter != frame_no:
                    f.write(',')
                else:
                    f.write(']')

            arr_timepoints.clear()

            baseTime = baseTime + 0.1

            frame_counter = frame_counter + 1

            # cv2 save frame as image
            name = "frame%d.jpg"%frame_counter
            cv2.imwrite(name, frame) 

            # out.write(frame)
    
        if frame_counter > frame_no:
            running = False
            # out.release()
            raise RuntimeError("No frame received")

def stop():
    """Stop loop and release camera.
    """
    running = False
    time.sleep(0.1)
    cap.release()

def process_frame(frame):
    try:
        results = detector.process(frame)
        return results
        
    except Exception as e:
        print(e)

if __name__ == '__main__':
    run()
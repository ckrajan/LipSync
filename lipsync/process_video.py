import time
import cv2
import numpy as np
import mediapipe as mp
import json
import ast
from ruamel.yaml import YAML
import os
from moviepy.editor import *

# cap = None

detector = mp.solutions.face_mesh.FaceMesh(
max_num_faces=1,
refine_landmarks=False,
min_detection_confidence=0.5,
min_tracking_confidence=0.5
)

arr_mouthPoints = []
arr_timepoints =[]
frame_no = 500

yaml = YAML(typ="safe")
with open("config.yml") as f:
    config = yaml.load(f)

movie_name = config["movie"]

def frame_collect(video,scenes,vid_type):
    clip = VideoFileClip(video)

    for i in scenes:
        # scene_path = "/home/chathushkavi/%s/%s/%s" % (movie_name,"output",i)
        scene_no = i
        os.mkdir("/home/chathushkavi/%s/%s/%s" % (movie_name,vid_type,i))
        os.mkdir("/home/chathushkavi/%s/%s/%s/%s" % (movie_name,vid_type,i,scene_no))

        scene_values = scenes[i]
        start_time = scene_values['start']
        end_time = scene_values['end']

        clip_x = clip.subclip(start_time, end_time)
        clip_x.write_videofile("/home/chathushkavi/%s/%s/%s/%s" % (movie_name,vid_type,i,scene_no) + '/' + i + ".mp4")
    
        """Start loop in thread capturing incoming frames.
        """
        cap = cv2.VideoCapture("/home/chathushkavi/%s/%s/%s/%s" % (movie_name,vid_type,i,scene_no) + '/' + i + ".mp4")

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # cap.get(3) is the video width, cap.get(4) is the video height.
        cap_width = int(cap.get(3))
        cap_height = int(cap.get(4))

        fps = cap.get(cv2.CAP_PROP_FPS)

        approx_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        approx_frame_count_x = approx_frame_count - 1
        print("Total frames: {}".format(approx_frame_count))
        print("FPS :",fps)
        running = True
        frame_counter = 0
        baseTime = 0.1

        # out = cv2.VideoWriter("/home/chathushkavi/Downloads/output.mp4", fourcc, fps, (cap_width, cap_height))

        while running:
            ret, frame = cap.read()
            frame_result = process_frame(frame)
            keypoints = frame_result.multi_face_landmarks

            if(keypoints != None):
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

                # with open('/home/chathushkavi/Downloadslipsync/target.json', 'a') as f:
                #     if frame_counter == 0:
                #         f.write('[')
                #     json.dump(res_list, f)
                #     if frame_counter != frame_no:
                #         f.write(',')
                #     else:
                #         f.write(']')

                arr_timepoints.clear()

                baseTime = baseTime + 0.1

                frame_counter = frame_counter + 1

                # cv2 save frame as image                
                name = "frame%d.jpg"%frame_counter
                cv2.imwrite(os.path.join("/home/chathushkavi/%s/%s/%s" % (movie_name,vid_type,scene_no) , name), frame)

                # out.write(frame)
            
            if(frame_counter > approx_frame_count_x):
                print(frame_counter,approx_frame_count_x)
                running = False
                # out.release()
                # raise RuntimeError("No frame received")

def run():
    try:
        os.mkdir("/home/chathushkavi/%s"%movie_name)
        # os.mkdir("/home/chathushkavi/%s/%s" % (movie_name,"input"))
        os.mkdir("/home/chathushkavi/%s/%s" % (movie_name,"output"))
        print ("Directory is created")
    except FileExistsError:
        print ("Directory already exists")

    input_video = config['input_video']
    input_scenes = config['input_scenes']
    output_video = config['output_video']
    output_scenes = config['output_scenes']

    # frame_collect(input_video,input_scenes,"input")
    frame_collect(output_video,output_scenes,"output")    

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
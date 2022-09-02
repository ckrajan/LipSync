import time
import cv2
import numpy as np
import mediapipe as mp
import json
import ast
from ruamel.yaml import YAML
import os
from moviepy.editor import *
import ast

mp_drawing = mp.solutions.drawing_utils

detector = mp.solutions.face_mesh.FaceMesh(
max_num_faces=10,
refine_landmarks=True,
min_detection_confidence=0.5,
min_tracking_confidence=0.5
)

mp_face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5)

frame_no = 10
arr_mouth = []
frame_counter = 1

outer_pts = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
inner_pts = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

yaml = YAML(typ="safe")
with open("config.yml") as f:
    config = yaml.load(f)

movie_name = config["movie"]
vid_type = "output"

def frame_collect(video,scenes,vid_type):
    clip = VideoFileClip(video)

    for i in scenes:
        scene_no = i
        os.mkdir("/home/chathushkavi/%s/%s/%s" % (movie_name,vid_type,i))
        os.mkdir("/home/chathushkavi/%s/%s/%s/%s" % (movie_name,vid_type,i,scene_no))
        os.mkdir("/home/chathushkavi/%s/%s/%s/%s" % (movie_name,vid_type,i,"facemesh"))
        os.mkdir("/home/chathushkavi/%s/%s/%s/%s" % (movie_name,vid_type,i,"detected_face"))
        os.mkdir("/home/chathushkavi/%s/%s/%s/%s" % (movie_name,vid_type,i,"output_scene"))

        scene_values = scenes[i]
        start_time = scene_values['start']
        end_time = scene_values['end']

        clip_x = clip.subclip(start_time, end_time)
        clip_x.write_videofile("/home/chathushkavi/%s/%s/%s/%s" % (movie_name,vid_type,i,scene_no) + '/' + i + ".mp4", fps=30)

        # clip_x.audio.write_audiofile("/home/chathushkavi/%s/%s/%s/%s" % (movie_name,vid_type,i,"output_scene") + '/output_' + i + "_audio.mp3")
    
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
        frame_counter = 1
        baseTime = 0.1

        out = cv2.VideoWriter("/home/chathushkavi/%s/%s/%s/%s"% (movie_name,vid_type,i,"output_scene") + '/output_' + i + "_video.mp4", fourcc, fps, (cap_width, cap_height))

        while running:
            ret, frame = cap.read()
            if frame is not None:

                face_cnt = 0
                cv2.imwrite("/home/chathushkavi/%s/%s/%s/%s/%s" % (movie_name,vid_type,scene_no,"detected_face","frame%d.jpg"%frame_counter), frame)
                image = cv2.imread(os.path.join("/home/chathushkavi/%s/%s/%s/%s" % (movie_name,vid_type,scene_no,"detected_face") , "frame%d.jpg"%frame_counter))
                image_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                results = mp_face_detection.process(image_input)

                if not results.detections:
                    print('No faces detected',frame_counter)

                    frame_result = process_frame(frame)
                    keypoints = frame_result.multi_face_landmarks

                    if(keypoints is not None):
                        ls_single_face=keypoints[0].landmark 
                        height, width, _ = image.shape
                        drawlips(frame,outer_pts,ls_single_face,height, width)
                        drawlips(frame,inner_pts,ls_single_face,height, width)
                        cv2.imwrite("/home/chathushkavi/%s/%s/%s/%s/%s" % (movie_name,vid_type,scene_no,"facemesh","frame%d.jpg"%frame_counter), frame)

                else:
                    height, width, _ = image.shape
                    for detection in results.detections: # iterate over each detection and draw on image
                        # mp_drawing.draw_detection(image, detection)
                        bbox = detection.location_data.relative_bounding_box
                        bbox_points = {
                            "xmin" : int(bbox.xmin * width),
                            "ymin" : int(bbox.ymin * height),
                            "xmax" : int(bbox.width * width + bbox.xmin * width),
                            "ymax" : int(bbox.height * height + bbox.ymin * height)
                        }
                        cropped_image = image[bbox_points["ymin"]:bbox_points["ymax"], bbox_points["xmin"]:bbox_points["xmax"]]

                        face_cnt = face_cnt + 1

                        height_crop, width_crop, _ = cropped_image.shape
                        
                        frame_result1 = process_frame(cropped_image)
                        keypoints1 = frame_result1.multi_face_landmarks
                        if(keypoints1 is not None):
                            ls_single_face1=keypoints1[0].landmark  

                            drawlips(cropped_image,outer_pts,ls_single_face1,height_crop, width_crop)
                            drawlips(cropped_image,inner_pts,ls_single_face1,height_crop, width_crop)

                            image[bbox_points["ymin"]:bbox_points["ymax"], bbox_points["xmin"]:bbox_points["xmax"]] = cropped_image

                    cv2.imwrite("/home/chathushkavi/%s/%s/%s/%s/%s" % (movie_name,vid_type,scene_no,"facemesh","frame%d.jpg"%(frame_counter)), image)

                out.write(image)

            frame_counter = frame_counter + 1   
            if(frame_counter > approx_frame_count_x):
            # if(frame_counter > frame_no):  
                running = False
                out.release()
                # raise RuntimeError("No frame received")

def run():
    try:
        os.mkdir("/home/chathushkavi/%s"%movie_name)
        os.mkdir("/home/chathushkavi/%s/%s" % (movie_name,"input"))
        print ("Directory is created")
    except FileExistsError:
        print ("Directory already exists")

    input_video = config['input_video']
    input_scenes = config['input_scenes']

    frame_collect(input_video,input_scenes,"input") 

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

def drawlips(frame,pts,ls_single_face,height, width):
    for i in pts:
        pt1 = ls_single_face[i]
        x = int(pt1.x * width)
        y = int(pt1.y * height)

        coordinates = [x, y]
        arr_mouth.append(coordinates)
        
        cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)
    
    # new points for polygon
    # create and reshape array
    arr_mouth1 = np.array(arr_mouth)
    arr_mouth1 = arr_mouth1.reshape((-1, 1, 2))

    # Attributes
    isClosed = True
    color = (255, 0, 0)
    thickness = 2

    # draw closed polyline
    cv2.polylines(frame, [arr_mouth1], isClosed, color, thickness)

    arr_mouth.clear()


if __name__ == '__main__':
    run()

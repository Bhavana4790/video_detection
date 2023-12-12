import argparse
import logging
import os
import json
from pathlib import Path
import types
import numpy as np
import cv2
import torch
import yt_dlp
from data_reader import InputType, get_all_files, get_input_type
from predictor import Predictor
from timm.utils import setup_default_logging
from moviepy.editor import VideoFileClip
import io
import tempfile
import xml.etree.ElementTree as ET
import csv 

_logger = logging.getLogger("inference")


def get_direct_video_url(video_url):
    ydl_opts = {
        "format": "bestvideo",
        "quiet": True,  # Suppress terminal output (remove this line if you want to see the log)
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)

        if "url" in info_dict:
            direct_url = info_dict["url"]
            resolution = (info_dict["width"], info_dict["height"])
            fps = info_dict["fps"]
            yid = info_dict["id"]
            return direct_url, resolution, fps, yid

    return None, None, None, None


def get_local_video_info(vid_uri):
    cap = cv2.VideoCapture(vid_uri)

    if not cap.isOpened():
        raise ValueError(f"Failed to open video source {vid_uri}")
    
    res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return res, fps


def num_frames(video_path):
    video_capture = cv2.VideoCapture(video_path.name)

    # Initialize a variable to count the frames
    frame_count = 0

    # Loop through the video frames and count them
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_count += 1

    return frame_count

# def dict_to_xml(data):
#     root = ET.Element('data')

#     for key, values in data.items():
#         item_elem = ET.SubElement(root, 'item', {'id': str(key)})

#         for item in values:
#             frame_elem = ET.SubElement(item_elem, 'frame', {'number': str(item['frame'])})

#             age_elem = ET.SubElement(frame_elem, 'age')
#             age_elem.text = str(item.get('age', ''))

#             gender_elem = ET.SubElement(frame_elem, 'gender')
#             gender_elem.text = str(item.get('gender', ''))

#             gender_score_elem = ET.SubElement(frame_elem, 'gender_score')
#             gender_score_elem.text = str(item.get('gender_score', ''))

#             bbox_elem = ET.SubElement(frame_elem, 'bbox')
#             bbox_elem.text = ', '.join(map(str, item.get('bbox', [])))

#             bbox_confidence_elem = ET.SubElement(frame_elem, 'bbox_confidence')
#             bbox_confidence_elem.text = str(item.get('bbox_confidence', ''))

#     return ET.tostring(root, encoding='utf-8', method='xml')

# def save_xml(data, file_path):
#     xml_content = dict_to_xml(data)

#     with open(file_path, 'wb') as xml_file:
#         xml_file.write(xml_content)

def save_dict_to_csv(data, file_path):
    with open(file_path, 'w', newline='') as csv_file:
        fieldnames = ['id', 'frame', 'age', 'gender', 'gender_score', 'bbox', 'bbox_confidence']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the data
        for key, values in data.items():
            for item in values:
                writer.writerow({
                    'id': key,
                    'frame': item['frame'],
                    'age': item.get('age', ''),
                    'gender': item.get('gender', ''),
                    'gender_score': item.get('gender_score', ''),
                    'bbox': ', '.join(map(str, item.get('bbox', []))),
                    'bbox_confidence': item.get('bbox_confidence', '')
                })


def do_demo(arguments):
    # Convert the dictionary to a SimpleNamespace
    dct = types.SimpleNamespace(**arguments)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    os.makedirs(dct.output, exist_ok=True)

    predictor = Predictor(config = dct, verbose=True)

    input_type = get_input_type(dct.input)

    if input_type == InputType.Video or input_type == InputType.VideoStream:
        if not dct.draw:
            raise ValueError("Video processing is only supported with --draw flag. No other way to visualize results.")

        if "youtube" in dct.input:
            dct.input, res, fps, yid = get_direct_video_url(dct.input)

            if not dct.input:
                raise ValueError(f"Failed to get direct video url {dct.input}")
            
            outfilename = os.path.join(dct['output'], f"out_{yid}.avi")
            
        else:
            bname = os.path.splitext(os.path.basename(dct.input))[0]
            outfilename = os.path.join(dct.output, f"out_{bname}.avi")
            res, fps = get_local_video_info(dct.input)
            print("frames per second: ",fps)

        # fps = fps+200
        if dct.draw:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            out = cv2.VideoWriter(outfilename, fourcc, fps, res)
            _logger.info(f"Saving result to {outfilename}..")

        j = 0
        for (detected_objects_history, frame) in predictor.recognize_video(dct.input):
            if dct.draw:
                out.write(frame)
            j += 1
            if j >= dct.num_frames:
                print(f"Stopping after {dct.num_frames} frames")
                break

        if dct.dump_history:
            history = {}
            for k, v in detected_objects_history.items():
                history[k] = [dict(zip(["frame", "age", "gender", "gender_score", "bbox", "bbox_confidence"],
                                       [_v.cpu().numpy().tolist()[0]
                                        if isinstance(_v, torch.Tensor) else _v
                                        for _v in vals]))
                              for vals in v]
        
        # save_xml(history, os.path.join(dct.output, 'output.xml'))
        save_dict_to_csv(history,os.path.join(dct.output, 'output.csv'))
        
        if len(history) == 0:
            return 0, 0
        else:

            # json_file = os.path.join(dct.output, f"out_{bname}.json")
            # with open(json_file, "w") as f:
            #     json.dump(history, f)
            return len(history), history


def main_func(input_path, frames):
    dct = {
        'input': input_path.name,
        'output': 'results/',
        'detector_weights': 'yolov8x_person_face.pt',
        'checkpoint': "mivolo_imbd.pth.tar",
        'with_persons': True,  # Set to True or False as needed
        'disable_faces': True,  # Set to True or False as needed
        'draw': True,  # Set to True or False as needed
        'dump_history': True,  # Set to True or False as needed
        'num_frames': frames,  # Set to the desired number of frames
        'device': 'cpu',  # Set to 'cuda' or 'cpu' as needed
    }
    setup_default_logging()
    count, result = do_demo(dct)
    print("Analysis finished")
    return count, result
    

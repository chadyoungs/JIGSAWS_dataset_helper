"""
Created on Thu Aug 27 14:43:42 2020
@author: xiaoxiaoyang
Update on Wed Nov 2 13:46:00 2023
"""

import cv2 
import os

from constant import ROOT, TASK
from metadata_generation import MetaData


video_root = os.path.join(ROOT, TASK, "video")
surgeme_video_root = os.path.join(ROOT, TASK, "surgeme_video")


def get_metadata():
    metadata = MetaData()
    metadata.generate_metadata()

    return metadata.metadata_res

def make_dirs(metadata):
    surgeme_list = metadata["metadata"]["surgeme_list"]
    for surgeme in surgeme_list:
        os.makedirs(os.path.join(surgeme_video_root, surgeme), exist_ok=True)

def video_surgeme_generation(metadata):
    capture1_video_list = [os.path.join(video_root, i) for i in os.listdir(video_root) if i.__contains__("capture1")]
    for capture1_video in capture1_video_list:
        trial_name = "_".join(os.path.basename(capture1_video).split("_")[:2])
        
        surgeme_start_end_frame = metadata[trial_name]["surgeme_start_end"]
        surgeme_start_frame_idx_list, surgeme_end_frame_idx_list, surgeme_list = surgeme_start_end_frame["start_frame_idx"], surgeme_start_end_frame["end_frame_idx"], surgeme_start_end_frame["surgeme"]
        cap = cv2.VideoCapture(capture1_video)
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        init, frame_count, surgeme_count = True, 1, 0

        if cap.isOpened():
            while init:
                success, frame = cap.read()
                if success:
                    if frame_count == surgeme_start_frame_idx_list[surgeme_count]:
                        videoWriter = cv2.VideoWriter(os.path.join(surgeme_video_root, surgeme_list[surgeme_count], "_".join([trial_name, surgeme_list[surgeme_count], str(surgeme_count)]) + ".avi"),
                                                      cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, (int(frame_width), int(frame_height))) 
                    
                    videoWriter.write(frame)

                    if frame_count == surgeme_end_frame_idx_list[surgeme_count]:
                        surgeme_count += 1
                    
                    if frame_count == surgeme_end_frame_idx_list[-1]:
                        init = False
                    
                    frame_count += 1
        
        cap.release()

def test():
    pass

def main():
    res = get_metadata()
    make_dirs(res)
    video_surgeme_generation(res)


if __name__ == "__main__":
    #main()
    pass

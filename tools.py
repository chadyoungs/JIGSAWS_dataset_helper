"""
Created on Wed Nov 2 13:46:00 2023
@author: xiaoxiaoyang
"""

import cv2
import numpy as np
import os
from PIL import Image

from constant import ROOT, TASK, IMG_HEIGHT, IMG_WIDTH, IMG_BLANK_WIDTH, STITCH_IMAGE_WIDTH


def frame_capture(gesture, capture_file_name, task=TASK):
    surgeme_video_dir = os.path.join(ROOT, task, "surgeme_video", gesture)
    surgeme_video_list = [os.path.join(surgeme_video_dir,i) for i in os.listdir(surgeme_video_dir) if i.endswith(".avi")]

    random = np.random.randint(len(surgeme_video_list))
    random_video = surgeme_video_list[random]
    
    init = True
    cap = cv2.VideoCapture(random_video)
    framenumber = int(cap.get(7))
    frame_count, selected_framenumber = 1, framenumber // 2 

    if cap.isOpened():
        while init:
            success, frame = cap.read()
            if success:
                if frame_count == selected_framenumber:
                    cv2.imencode('.jpg', frame)[1].tofile(capture_file_name)
                    init = False
                frame_count += 1

        cap.release()


def extract_frame():
    video_dataset_root = os.path.join(ROOT, TASK, "surgeme_video")
    video_dirs = [os.path.join(video_dataset_root, video_dir) for video_dir in os.listdir(video_dataset_root)]
    for video_dir in video_dirs:
        videos = [os.path.join(video_dir, video) for video in os.listdir(video_dir)]
        for video in videos:
            cap = cv2.VideoCapture(video)
            count, init = 0, True
            if cap.isOpened():
                while init:
                    success, frame = cap.read()
                    if success:
                        img_folder = os.path.join(ROOT, TASK, "surgeme_img", os.path.basename(os.path.dirname(video)), os.path.basename(video))
                        img_file_name = "img_{:0>5}.jpg".format(count)
                        os.makedirs(img_folder, exist_ok=True)
                        cv2.imwrite(os.path.join(img_folder, img_file_name), frame)
                    else:
                        init = False

                    count += 1
            cap.release()


def image_stitch(input_img_list, stitch_img, quantity_value=100):
    stitch_img = Image.new('RGB',(STITCH_IMAGE_WIDTH, IMG_HEIGHT))

    left = 0
    right = IMG_WIDTH
    for count, image in enumerate(input_img_list):
        # img
        stitch_img.paste(Image.open(image), (left, 0, right, IMG_HEIGHT))
        
        # img blank
        left += IMG_WIDTH
        right += IMG_WIDTH
        stitch_img.paste((255, 255, 255), (left, 0, left+IMG_BLANK_WIDTH, IMG_HEIGHT))
        if count == 2:
            break
        left += 10
        right += 10
    
    stitch_img.save('image_stitch.jpg', quantity = quantity_value)


if __name__ == "__main__":
    #frame_capture('G5', 'test.jpg')
    extract_frame()
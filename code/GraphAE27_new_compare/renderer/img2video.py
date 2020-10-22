#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 17:59:51 2018

@author: yi
"""

#!/usr/local/bin/python3

import cv2
import argparse
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
args = vars(ap.parse_args())

# Arguments

ext = args['extension']



# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case


frame_ranges = [(0,256),(256,512),(512,768),(768,1024)]
for v in range(24,48):
    for h in range(0,2):
        dir_path = '/home/zhouyi/Documents/MotionGAN/training/test5301_motionInterGAN21/test/user_study/motion_clips/%03d'%v+"_"+str(h)+'/'
        out_path = '/home/zhouyi/Documents/MotionGAN/training/test5301_motionInterGAN21/test/user_study/motion_clips/%03d'%v+"_"+str(h)
        # Determine the width and height from the first image
        image_path =  os.path.join(dir_path,  "%02d"%0+".png")
        frame = cv2.imread(image_path)
        cv2.imshow('video',frame)
        height, width, channels = frame.shape
        for (start_id,end_id) in frame_ranges:
            output=out_path+"_%05d"%start_id+".mp4"
            out = cv2.VideoWriter(output, fourcc, 60.0, (width, height))
            
            for i in range(start_id, end_id):
                image_path = os.path.join(dir_path,  "%02d"%i+".png")
                frame = cv2.imread(image_path)
            
                out.write(frame) # Write out frame to video
        
                cv2.imshow('video',frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
                    break
            out.release()
            cv2.destroyAllWindows()
            print("The output video is {}".format(output))

# Release everything if job is finished



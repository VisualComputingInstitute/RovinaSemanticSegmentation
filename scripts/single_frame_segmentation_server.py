#!/usr/bin/env python

from semantic_segmentation.srv import *
from cv_bridge import CvBridge
import rospy

import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

def segment_frame(req):
    rgb = br.imgmsg_to_cv2(req.rgb)
    depth = br.imgmsg_to_cv2(req.depth)

    #cv2.imshow('d0', depth[:,:,0]/5.0)
    #cv2.imshow('d1', depth[:,:,1]/5.0)
    #cv2.imshow('d2', depth[:,:,2]/5.0)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #TODO finish semantic segmentation.

    #Temporary segmentation
    probability_list = []
    for c in class_counts:
        probabilities = np.zeros((c,rgb.shape[0],rgb.shape[1]), dtype=np.float32)

        #Init probabilities
        probabilities[:3,:,:] = np.ones((3,rgb.shape[0],rgb.shape[1]), dtype=np.float32)/3.0

        #Find the points below a certain height and decide on the class.
        probabilities[0, depth[:,:,2] > 0]   = 1.0 # floor class
        probabilities[1, depth[:,:,2] > 0]   = 0.0
        probabilities[2, depth[:,:,2] > 0]   = 0.0

        probabilities[0, depth[:,:,2] > 0.5] = 0.0
        probabilities[1, depth[:,:,2] > 0.5] = 1.0 # above 0.5 overwrite with wall
        probabilities[2, depth[:,:,2] > 0.5] = 0.0

        probabilities[0, depth[:,:,2] > 1.5]   = 0.0
        probabilities[1, depth[:,:,2] > 1.5]   = 0.0
        probabilities[2, depth[:,:,2] > 1.5]   = 1.0 # above 3 overwrite with ceiling.

        probability_list.append(probabilities)
        #cv2.imshow('ceiling', (rgb*probabilities[2][...,None]).astype(np.uint8))
        #cv2.imshow('wall', (rgb*probabilities[1][...,None]).astype(np.uint8))
        #cv2.imshow('floor', (rgb*probabilities[0][...,None]).astype(np.uint8))
        #cv2.waitKey(1)
        #cv2.destroyAllWindows()

    out = np.concatenate([p.swapaxes(0,2).swapaxes(0,1).flat for p in probability_list], axis=0)
    return SingleFrameSegmentationResponse(out)

def single_frame_segmentation_server():
    rospy.init_node('single_frame_segmentation_server')
    s = rospy.Service('/semantic_segmentation/SingleFrameSegmentation', SingleFrameSegmentation, segment_frame)

    #Parse the layer information
    ns = rospy.get_name() + '/'
    with open(rospy.get_param(ns + 'config_file')) as config_file:
        config = json.load(config_file)

    global layer_count, layer_names, class_counts
    layer_count = 0
    layer_names = []
    class_counts = []

    for coding in config['color_codings']:
        layer_count +=1
        layer_names.append(coding['name'])
        class_counts.append(len(coding['coding'])-1)

    #Get the neural network and see if the layer sizes match.
    if False:
        raise RuntimeError('There is a mismatch of sizes bla bla TODO')

    print('SingleFrameSegmentation server ready!')
    global br
    br = CvBridge()
    rospy.spin()

if __name__ == '__main__':
    single_frame_segmentation_server()
#!/usr/bin/env python
'''
Simple image viewer and capturer:
 rosrun lab4_color im_capture.py        : subscribes to /raspicam_not/image/compressed 

Complete functions as indicated

'''
import os
import time
import argparse
import rospy
import cv2
from dnn_detect import Dnn
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class im_show:
    
    def __init__(self, args,data):
        self.bridge = CvBridge()
        rospy.Subscriber('/cam_front/raw', Image, self.display)
        rospy.loginfo('Image capture from raspberry topic.  Press "s" to save, "q" to quit.')
        self.dnn_f = Dnn(args,data)

    def display(self, msg):
        ''' Uses OpenCV to display an image and waits 2ms
            If the user presses 's', then calls save() with the image
            If the user presses 'q', then quits 
        '''
        # complete here
        try:
            
            img = self.bridge.imgmsg_to_cv2(msg,'bgr8')
            
        except CvBridgeError as e:
            print(e)
            rospy.signal_shutdown('Quitting')

        start = time.time()
        classes, scores, boxes = self.dnn_f.detect(img)
        end = time.time()

        start_drawing = time.time()
        self.dnn_f.draw(img, classes, scores, boxes)
        end_drawing = time.time()
    
        fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
        cv2.putText(img, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow("detections", img)

        
        wait = cv2.waitKey(2)
        if wait & 0xFF == ord('q'):
            rospy.signal_shutdown('Quitting')


  

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Image Capture from ros topic')
    parser.add_argument('folder', type=str,    help='Folder to save images in')
    #parser.add_argument('nms', type=str,    help='--')
    parser.add_argument('threshold',   default = 0.4,  type=float,      help='data')
    args, unknown = parser.parse_known_args()  # For roslaunch compatibility
    if unknown: print('Unknown args:',unknown)

    rospy.init_node('im_capture')  # Start a ROS node
    ic = im_show( args.folder,args.threshold)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()

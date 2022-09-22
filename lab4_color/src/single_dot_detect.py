#!/usr/bin/env python
'''
Detect a single target
 rosrun lab4_color single_single_dot_detect.py imgname maskname 

A logistic regression classifier for targets is created from:
 imgname: name of saved image
 maskname: name of mask for saved image indicating which pixels are target pixels
'''
import argparse
import rospy
import cv2
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from logist_reg import LogisticReg, plotTargets  # Use this for classification and plotting

class SingleDotDetect:
    
    def __init__(self, logr):
        ''' Initialize your node and create a subscriber to: 'raspicam_node/image/compressed'
            This should make self.detect() the callback function
        '''
        # Complete this
        rospy.init_node('ImageSubscriber')
        self.bridge = CvBridge()
        #topic_name = 'raspicam_node/image/compressed'
        rospy.Subscriber('raspicam_node/image/compressed', CompressedImage, self.detect)
        self.reg = logr


    def detect(self, img_msg):
        ''' This is the image callback.  It should read in the image to an OpenCV image and
            then use find_largest_target() from LogisticReg to find the largest target
            It should also plot the detected centroid using plotTargets()
        '''
        # Complete this
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(img_msg,'bgr8')
        except CvBridgeError as e:
            print(e)
        
        #testimg = cv2.imread(img)
        probt = self.reg.apply( img )
        
        centroid, area, target_mask = self.reg.find_largest_target(probt)
        rate = rospy.Rate(20)
        rate.sleep()
        # Plot detected region and centroid:
        plotTargets(img, target_mask, [centroid] )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown('Quitting')



        




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image viewer')
    parser.add_argument('imagepath', type=str, metavar='imagepath', help='Full pathname of image')
    parser.add_argument('maskpath', type=str, metavar='maskpath', help='Full pathname of mask')
    args, unknown = parser.parse_known_args()  # For roslaunch compatibility
    if unknown: print('Unknown args:',unknown)

    # Start by buiding logistic regression model
    lr = LogisticReg() 
    lr.fit_model_to_files( args.imagepath, args.maskpath )
    lr.print_params()

    dd = SingleDotDetect( lr )
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()


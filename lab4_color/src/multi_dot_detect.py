#!/usr/bin/env python
'''
Detects multiple targets and publishes their centroids

 rosrun lab4_color multi_dot_detect.py imgname maskname 

A logistic regression classifier for targets is created from:
 imgname: name of saved image
 maskname: name of mask for saved image indicating which pixels are target pixels

'''
import argparse
import rospy
import cv2
from std_msgs.msg import Header
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from lab4_color.msg import Point32Array
from cv_bridge import CvBridge, CvBridgeError
from logist_reg import LogisticReg 

class MultiDotDetect:
    
    def __init__(self, logr):
        ''' Initialize detector that:
            Subscribes to: 'raspicam_node/image/compressed'
            Creates a publisher to: 'dots' of type Point32Array 
        '''
        rospy.init_node('multi_dot_detect', anonymous=True)
        self.bridge = CvBridge()
        self.current = []
        self.logr = logr
        rospy.Subscriber('raspicam_node/image/compressed', CompressedImage, self.detect)
        self.pub = rospy.Publisher('dots', Point32Array, queue_size=1)
        rospy.loginfo('Publishing detections to /dots')

    def detect(self, img_msg):
        ''' Read in an image and detects multiple target centroids on it
            Publishes centroids of detected targets
            Make sure to fill the header of the target centroid message with the image timestamp
        '''
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(img_msg,'bgr8')
        except CvBridgeError as e:
            print(e)
        
        prob_target = self.logr.apply(img)
        centroids, _, _ = self.logr.find_all_targets(prob_target, threshold=0.5, minpix=20)        

        self.publish_centroids( centroids, img_msg.header )

    def publish_centroids(self, centroids, imgheader):
        ''' Publishes centroids of detected targets '''
        pa = Point32Array(header=imgheader)  #copy header from image
        for cen in centroids:
            pa.points.append(Point(cen[0],cen[1],0))
        self.pub.publish( pa )


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

    dd = MultiDotDetect( lr )
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()

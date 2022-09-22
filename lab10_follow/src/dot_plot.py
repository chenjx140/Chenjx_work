#!/usr/bin/env python
'''
Detects multiple targets and publishes their centroids

 rosrun lab4_color plot_multi_dot.py
'''
import rospy
import cv2
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Point
from lab4_color.msg import Point32Array
from cv_bridge import CvBridge, CvBridgeError
from logist_reg import LogisticReg, plotTargets
import message_filters

class PlotMultiDot:
    
    def __init__(self):
        ''' This plotter needs to subscribe to both: 'raspicam_node/image/compressed' and 'dots'
            It should call plot_detections() with these messages
        '''
        rospy.init_node('PlotMultiDot', anonymous=True)
        self.bridge = CvBridge()
        imsub = message_filters.Subscriber('camera/image/compressed', CompressedImage)
        dotsub = message_filters.Subscriber('dots', Point32Array)
        self.ts = message_filters.TimeSynchronizer([imsub, dotsub], queue_size=1)
        self.ts.registerCallback(self.plot_detections)        
        rospy.loginfo('Plotting detected targets')

    def plot_detections(self, img_msg, dot_msg):
        ''' Plots target centroids on top of corresponding image
            Quit node if user presses 'q'        
        '''
        try:
            img = self.bridge.compressed_imgmsg_to_cv2(img_msg,'bgr8')
        except CvBridgeError as e:
            print(e)
        centroids = []
     
        for pt in dot_msg.points:
            
            centroids.append( [pt.x,pt.y] )        

        plotTargets(img, [], centroids )
        if (cv2.waitKey(2) & 0xFF) == ord('q'):
            rospy.signal_shutdown('Quitting')

if __name__ == '__main__':

    PlotMultiDot()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()

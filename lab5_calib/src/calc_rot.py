#!/usr/bin/env python
import rospy
import cv2

from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Point32
from lab4_color.msg import Point32Array
import numpy as np
from scipy.spatial.transform import Rotation as R

class calibration:
    def __init__(self):
        rospy.init_node('calibrationSubscriber')
        rospy.Subscriber('/rays', Point32Array, self.detect)
        #self.pub = rospy.Publisher('/cam_rot', Point32Array, queue_size=10)

    def detect(self, rays):
        p2 = np.array([rays.points[0].x,rays.points[0].y,rays.points[0].z])
        p3 = np.array([rays.points[2].x,rays.points[2].y,rays.points[2].z])
        p1 = np.array([rays.points[1].x,rays.points[1].y,rays.points[1].z])
        
        cmx = - p2/np.linalg.norm(p2)

        cmz = np.cross(p1,p3)/(np.linalg.norm(np.cross(p1,p3)))

        cmy = np.cross(cmz,cmx)

        ro = np.column_stack((cmx,cmy,cmz))

        mR = ro.T
       
        r = R.from_matrix(mR)

        qu = r.as_quat().tolist()



      
        rospy.set_param('/cam_rot',qu)










    def publish(self,rays,ray_message):
        pass



if __name__=="__main__":
    calibration()
    rospy.spin()

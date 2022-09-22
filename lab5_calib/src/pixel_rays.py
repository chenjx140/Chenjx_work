#!/usr/bin/env python
import rospy
import cv2

from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Point32
from lab4_color.msg import Point32Array
import numpy as np

class Rays:
    def __init__(self):
        rospy.init_node('RaySubscriber')
        rospy.Subscriber('/dots', Point32Array, self.detect)
        self.pub = rospy.Publisher('/rays', Point32Array, queue_size=10)

    def detect(self, img_msg):
        ray_message = Point32Array()
        ray_message.header.frame_id = 'camera_pose'
        cam_msg = rospy.wait_for_message('/raspicam_node/camera_info', CameraInfo)
        D = np.array(cam_msg.D)
        K = np.array(cam_msg.K).reshape( (3,3) )

        point_matrix = np.zeros((len(img_msg.points),1,2),dtype=float)

        for i in range(0,len(img_msg.points)):
            point_matrix[i][0][0] = img_msg.points[i].x
            point_matrix[i][0][1] = img_msg.points[i].y

        u_flot = cv2.undistortPoints(point_matrix, K, D, R=None, P=None)

        rays = []
        for i in u_flot:
            rays.append(Point32(i[0][0],i[0][1],1))
        self.publish(rays,ray_message)



    def publish(self,rays,ray_message):
        rate = rospy.Rate(14)  
        while not rospy.is_shutdown():  # Exit loop if Ctrl-C pressed
            ray_message.points = rays

            self.pub.publish(ray_message)
            rate.sleep()



if __name__=="__main__":
    Rays()
    rospy.spin()


        









#!/usr/bin/env python
''' ground_spots.py
    Subscribes to `rays` topic which are assumed to correspond to points
    on the ground camera_pose coordinates.  Then given camera extrinsics, 
    this function calculates the 3D location of the ground points and
    publishes them a PointCloud2 topic called `ground_spots`

    Complete unfinished portions of this.
'''
import rospy
import numpy as np
import cv2
import PyKDL
from std_msgs.msg import Header
from geometry_msgs.msg import Point, TransformStamped, Transform
from lab4_color.msg import Point32Array
from sensor_msgs.msg import PointCloud2, PointField
from transform_frames import TransformFrames
from tf2_geometry_msgs import transform_to_kdl
from sensor_msgs import point_cloud2

def transform_point_array(pointarray, transform):
    ''' Apply a transform of type TransformStamped to each point in pointarray of type Point32Array '''
    for point in pointarray.points:
        p = transform_to_kdl(transform) * PyKDL.Vector(point.x, point.y, point.z)
        point.x = p[0]
        point.y = p[1]
        point.z = p[2]

class ground_spots():
    def __init__(self):
        rospy.init_node('ground_spots')

        # Use TransformFrames to find rotation and translation
        # of camera_pose with respect to base_footprint
        # ...
        self.tran = TransformFrames()
        # Store a transformation as a TransformStamped that is just the rotation component of this
        # ...
                        
        self.pub = rospy.Publisher('ground_spots', PointCloud2, queue_size=1)     
        rospy.Subscriber('rays', Point32Array, self.calc_spots)

        rospy.loginfo( 'Finding ground spots' )

    def calc_spots(self, ray_msg):
        ''' Calculate and publish spots on ground '''
        # First rotate into an axis aligned with base_footprint:
        rot = np.array([-0.1,0., 0.15])
        rot = rot.T
        # Use transform_point_array() to rotate points from camera to a base_footprint-aligned axis 
        # ...
        #FrameTransformer = self.tran

        transform = self.tran.get_transform('camera_pose', 'base_footprint')

        transform_point_array(ray_msg,transform)

        #print(ray_msg)
        Pb = 0
        lit = []
        for i in ray_msg.points:
            lamda = (-rot[2]/i.z)
            Pb = rot+lamda*np.array([i.x,i.y,i.z]).T
            i.x = Pb[0]
            i.y = Pb[1]
            i.z = Pb[2]




        self.publishToPointCloud( ray_msg)
        # Rescale points along rays so that they intersect the ground, and shift them by
        # the camera position in base_footprint:
        # ...

        # Publish points to a PointCloud2 using publishToPointCloud
        # ...


    def publishToPointCloud(self, ptarray):
        ''' Publish Point32Array in a PointCloud2 '''
        
        pcheader = Header(stamp=ptarray.header.stamp,frame_id='base_footprint')
        pts = []
        for pt in ptarray.points:
            pts.append([pt.x, pt.y, pt.z, 0])

        fields = [PointField('x', 0, PointField.FLOAT32,1),
                  PointField('y', 4, PointField.FLOAT32,1),
                  PointField('z', 8, PointField.FLOAT32,1),
                  PointField('intensity', 12, PointField.UINT32,1)]
        msg_PointCloud2 = point_cloud2.create_cloud(pcheader, fields, pts)
        self.pub.publish( msg_PointCloud2 )


if __name__=='__main__':
   ground_spots()
   rospy.spin()


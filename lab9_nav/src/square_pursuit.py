#!/usr/bin/env python
''' Coordinate Transform Demo
    Transforms a laser scan point from base_scan to odom

    Use of this is explained in AV_Notes/ROS/Coordinate_Transforms.md

'''
import argparse

from sympy.geometry.line import Segment
import rospy
import sympy
from sympy.geometry import Ray, Circle, Segment, intersection


from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist,PoseArray,Point, TwistStamped, PointStamped
import numpy as np
from transform_frames import TransformFrames

class square():
    def __init__(self, distance,speed):
        self.distance = distance
        self.speed = speed 
        self.move = Twist()
        rospy.init_node('square')
        self.trans = TransformFrames()
        self.pub_ps = rospy.Publisher('point_ahead', PointStamped, queue_size=1)    
        self.pub_ts = rospy.Publisher('cmd', TwistStamped, queue_size=1)    
        self.pub = rospy.Publisher('cmd_vel',Twist, queue_size=1)     
        rospy.Subscriber("/odom", Odometry, self.callback_odom,queue_size=1)
        
        
    def calculation(self,x,y):
        c = Circle((x,y),self.distance)
        down = Segment((0,0),(1,0))
        up = Segment((1,1),(0,1))
        left = Segment((0,1),(0,0))
        right = Segment((1,0),(1,1))
        
        return intersection(c,down)+intersection(c,up)+intersection(c,left)+intersection(c,right)
        
    def callback_odom(self, msg):
        x = msg.pose.pose.position.x
        y= msg.pose.pose.position.y
        
        inter = self.calculation(x,y)
        
        new_pts, header = self.transform_points(inter,msg.header,'base_footprint')
        
        new_pts.sort(key = lambda x:x[0],reverse = True )
        poin = new_pts[0]
        current = (poin[0],poin[1])
        
        k = (2*current[1])/(current[0]**2+current[1]**2)
        self.move.linear.x = self.speed
        self.move.angular.z = k*self.speed
        points = PointStamped(header = header, point = Point(poin[0],poin[1],0))
        self.pub_ps.publish(points)
        ts = TwistStamped(header = msg.header, twist = self.move)
        self.pub_ts.publish(ts)
        self.pub.publish(self.move)
        
        
        
       

    def transform_points(self, inter, header, new_frame):
        ''' Transform point coordinates into new_frame '''
        pts = []
        for i in inter:
            pts.append([i[0],i[1]])
        # Convert points to PoseArray (Note: this is *not* efficient for many points, just a few)
        pa = PoseArray(header=header) # Header specifies which frame points start in
        for p in pts:
            pose = Pose(position=Point(p[0],p[1],0))
            pa.poses.append(pose)
        # Call coordinate transform into new_frame:
        tran_pa = self.trans.pose_transform(pa, new_frame )
        # Convert PoseArray back into point list:
        new_pts = []
        for p in tran_pa.poses:
            new_pts.append( [p.position.x, p.position.y] )  # For now ignoring intensity value

        return new_pts, tran_pa.header  # Return transformed points and new header 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coordinates')
    parser.add_argument('--distance',    default=0.4, type=float,    help='Centroid publish frame')
    parser.add_argument('--speed',    default=0.2, type=float,    help='Centroid publish frame')
    args, unknown = parser.parse_known_args()  # For roslaunch compatibility
    
    if unknown: print('Unknown args:',unknown)
    square(args.distance, args.speed)
    rospy.spin()

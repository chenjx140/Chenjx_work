#!/usr/bin/env python
''' Coordinate Transform Demo
    Transforms a laser scan point from base_scan to odom

    Use of this is explained in AV_Notes/ROS/Coordinate_Transforms.md

'''
import argparse
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, Point, PoseArray
import numpy as np
from transform_frames import TransformFrames

class Centroid():
    def __init__(self, frame):
        self.pub_frame = frame
        rospy.init_node('centroid')
        self.trans = TransformFrames()
        self.pub = rospy.Publisher('/centroid', PointCloud2, queue_size=1)     
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)  
        rospy.spin()

    def lidar_callback(self, msg):
        T = 0.25               
        ranges = np.array(msg.ranges)           # Convert to Numpy array for vector operations
        angles = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min # Angle of each ray
        fir_n = ranges[0]
        end_n = ranges[-1]
        discov = []
        cluster = []
        for i in range(len(ranges)):
            if ranges[i] == np.inf:
                continue
            if i in discov:
                continue
            each = []
            if i ==  0:
                left_ = end_n
            else:
                left_ = ranges[i-1]

            while abs(left_ - ranges[i]) < T and ranges[i] != np.inf:
                if i == 0:
                    each.append(len(ranges)-1)
                each.append(i)
                discov.append(i)
                
                
                i += 1
                left_ = ranges[i-1]
                
                if i == 360:
                    break
            if len(each) != 0:
                cluster.append(each)

           



                





        #good = ranges < np.inf                  # Only finite returns are good

        for i in cluster:

            x = ranges[i] * np.cos(angles[i]) # vector arithmatic is much faster than iterating
            y = ranges[i] * np.sin(angles[i])
            raw_centroid = [[x.mean(),y.mean(),0.,0]]   # Centroid of object, in base_scan coordinates

            # Transform to a different coordinates, as specified by self.pub_frame:
            centroid, header = self.transform_points( raw_centroid, msg.header, self.pub_frame )

            # Output the frame we are describing the centroid in, and its coordinates:
            rospy.loginfo(f'frame: {header.frame_id}, centroid: {centroid[0][0]:.2f}, {centroid[0][1]:.2f}, {centroid[0][2]:.2f}')

            fields = [PointField('x', 0, PointField.FLOAT32,1),
                        PointField('y', 4, PointField.FLOAT32,1),
                        PointField('z', 8, PointField.FLOAT32,1),
                        PointField('intensity', 12, PointField.UINT32,1)]
            centroid_pc2 = point_cloud2.create_cloud(header, fields, centroid)

            self.pub.publish(centroid_pc2)

    def transform_points(self, pts, header, new_frame):
        ''' Transform point coordinates into new_frame '''
        # First see if we are actually changing frames:
        if header.frame_id == new_frame:
            return pts, header  # If not, then no need to transform anything

        # Convert points to PoseArray (Note: this is *not* efficient for many points, just a few)
        pa = PoseArray(header=header) # Header specifies which frame points start in
        for p in pts:
            pose = Pose(position=Point(p[0],p[1],p[2]))
            pa.poses.append(pose)
        # Call coordinate transform into new_frame:
        tran_pa = self.trans.pose_transform(pa, new_frame )
        # Convert PoseArray back into point list:
        new_pts = []
        for p in tran_pa.poses:
            new_pts.append( [p.position.x, p.position.y, p.position.z, 0] )  # For now ignoring intensity value

        return new_pts, tran_pa.header  # Return transformed points and new header 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coordinates')
    parser.add_argument('--frame',    default='odom', type=str,    help='Centroid publish frame')
    args, unknown = parser.parse_known_args()  # For roslaunch compatibility
    if unknown: print('Unknown args:',unknown)

    Centroid(args.frame)

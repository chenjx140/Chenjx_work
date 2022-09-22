#!/usr/bin/env python
'''
    squareDrive.py 

    This is an exmple ROS node for driving a Turtlebot in a circular arc and then stopping.

    Daniel Morris, Nov 2020
'''
import argparse
import rospy
from geometry_msgs.msg import Twist

class SqDrive:
    def __init__(self):
        rospy.init_node('move')
        self.pub = rospy.Publisher('/cmd_vel', Twist, latch=True, queue_size=1)
        
        self.move = Twist() 
            # latch: keeps published topic current until next published topic
	
    def start(self, linear): 
                        # Initialized all-zero values
        self.move.linear.x = linear         # Linear velocity in the x axis
        #move.angular.z = angular       # Angular velocity in the z axis
        self.pub.publish(self.move)  

    def steer(self,angular): 
        self.move.angular.z = angular
        #rospy.sleep(4) # sleep
       
        self.pub.publish(self.move)  

        rospy.sleep(1.05)

        self.move.angular.z = 0

        self.pub.publish(self.move)  



    def drive(self, time=2.0, linear=0.4, angular=1.5):
        info = f'Driving for sqaure drive'

        rospy.loginfo(info)

        rospy.sleep(1)
        self.start( 0.4)
        rospy.sleep( 2.145)
        #rospy.sleep( 3.292)
        self.steer(1.51)
        rospy.sleep( 1.40)

        self.steer(1.497)
        rospy.sleep( 1.3)

        self.steer(1.48)
        rospy.sleep( 1.4)
       


        
        rospy.sleep(10) # sleep
        self.stop()
        rospy.sleep( 0.1 )     # Ensure publishing completes before quitting
        rospy.loginfo('Done driving')

if __name__== '__main__':
    rospy.sleep(1)
    parser = argparse.ArgumentParser(description='Circular Arc Arguments')
    parser.add_argument('--time',    default=15, type=float,    help='Time to drive')
    parser.add_argument('--linear',  default=0.0, type=float,    help='Linear speed')
    parser.add_argument('--angular', default=0.0, type=float,    help='Angular speed')
        

    args, unknown = parser.parse_known_args()  # For roslaunch compatibility





    if unknown: print('Unknown args:',unknown)

    av = SqDrive()

    
    av.drive(args.time, args.linear, args.angular)

#!/usr/bin/env python
''' follower.py

    This is a ROS node that subscribes to keyboard commands and 
    moves the robot according to these.

    Your task is to complete the missing portions that are marked with
    # <...> 
'''
import rospy
import numpy as np
import cv2
from std_msgs.msg import Int32
from robot_room import RobotRoom

class Follower():
    def __init__(self, topic_name='robot_move'):
        ''' Initialize a ROS node.  Permit any number of follower nodes
            Define an variable for a room, but don't initialize the room (see below for reason)
            Define a subscriber to the 'robot_move' topic
        '''
        rospy.init_node("robfor") 
        
        self.bob2 = None
        rospy.Subscriber(topic_name, Int32, self.move_callback)

    def move_callback(self, msg):
        ''' Initialize a RobotRoom if one has not already been initialized
                Putting the initializer here ensures RobotRoom calls all remain in a single thread
                which is important since they use OpenCV
            If key is a 'q', shut down node with: rospy.signal_shutdown("Received 'q' message")
            Move the robot according to the key in the message
            Draw the RobotRoom
        '''
        if(self.bob2 == None):
            self.bob2 = RobotRoom("robo")
        
        
        
        move = msg.data

        if(move == ord('q')):
            rospy.signal_shutdown("Received 'q' message")
        self.bob2.move(move)
        #print(self.bob2.xy)
        self.bob2.draw()
        
        
if __name__ == '__main__':

    f = Follower()
    rospy.spin()


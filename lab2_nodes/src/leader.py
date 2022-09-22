#!/usr/bin/env python
''' leader.py

    This is a ROS node that defines room-A and robot-A. Robot-A is controlled 
    by the keyboard. Additionally the motion commands are published so other 
    nodes can follow.

    Your task is to complete the missing portions that are marked with
    # <...> 
'''
import rospy
from std_msgs.msg import Int32
from robot_room import RobotRoom

class Leader():
    
    def __init__(self, topic_name='robot_move'):
        ''' Initialize Node -- we only want one leader to run 
            Define a publisher for a message of type Int32
            Initialize a RobotRoom
            Call listKeys() to tell user what to press
        '''
        rospy.init_node("robotleader") 
        self.pub = rospy.Publisher(topic_name, Int32, queue_size=10)
        self.bob = RobotRoom("robot",(128,0,0))
        self.bob.draw()
        self.bob.listKeys()
        

    def run(self):
        ''' Create a loop that draws the robot, and based on the key pressed moves it accordingly 
            and publishes the key to the topic 
            Quit when the user presses a 'q'
        '''
        pstring = " "
        while pstring != 'q':
            pstring = input()

            self.bob.move(ord(pstring))
            
            
            #print(ord(pstring))
            self.pub.publish(ord(pstring))
            self.bob.draw()
            #print(self.bob.xy)
            if pstring == 'q':
                break


        
        
if __name__ == '__main__':
    lead = Leader()
    lead.run()

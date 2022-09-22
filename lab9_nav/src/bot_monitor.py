#!/usr/bin/env python
import matplotlib.pyplot as plt #plots

import rospy

from nav_msgs.msg import Odometry
class odemSub:

    def __init__(self, node_name='odem_char', topic_name='odom'):
        rospy.init_node(node_name) # Initialize ROS node
        # Create a subscriber that reads an Int32 from the topic and sends this to self.read_callback
        rospy.Subscriber(topic_name, Odometry, self.read_callback)
        self.x = 0 #0
        self.y = 0
        self.flag = False
        

    def read_callback(self, msg):
        ''' Decodes a single unicode character and outputs it to the terminal '''
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        #print(msg.pose.pose.position.x)
        #print(msg.pose.pose.position.y)
        if self.flag == False:
            plt.grid(True, linestyle='--')
            plt.gca().set_aspect('equal', 'box')
            self.flag = True
        #plt.axis([0, 2.0, 0, 2.0])

        
        plt.show(block=False)
        
    
        
        if((abs(x - self.x) > 0.05)  or ( abs(y - self.y) >0.05)):
            self.x = x
            self.y = y

            plt.plot(self.x,self.y,'ro')
            plt.gcf().canvas.flush_events()
            plt.show(block=False)
            plt.show(block=False)
            rospy.sleep(0.1)




            

        
        

        

        
        
        


        

if __name__=="__main__":
    
    mydecoder = odemSub()  # Create a subscriber that reads and decodes the topic

 

    
    
    

    rospy.spin() # Wait and perform callbacks until node quits

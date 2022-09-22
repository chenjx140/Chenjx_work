#!/usr/bin/env python
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from lab4_color.msg import Point32Array
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
from pid import pid_controller
import matplotlib.pyplot as plt
import argparse

class follow():

    def __init__(self, speed, kp, ki, kd, use_rate=False):

        self.speed = speed
        self.use_rate = use_rate
        self.pid = pid_controller(kp, ki, kd)

        self.target_yaw_list = []
        self.meas_yaw_list = []
        
        self.cmd_yaw_rate_list = []
        
        self.time_list = []

        rospy.init_node('pid_test')        
        self.pub_vel = rospy.Publisher('/cmd_vel', Twist, latch=True, queue_size=1)

        rospy.Subscriber('/dots', Point32Array, self.callback_odom, queue_size=1)  # Important to have queue size of 1 to avoid delays
     

    def callback_odom(self, msg):
        self.time_list.append(rospy.Time.now().to_sec())
        #print(msg.points[0].x)
        dtime = self.time_list[-1] - self.time_list[0]
        if msg.points is not None:
            target_yaw = msg.points[0].x

        current_yaw  = 160
        self.meas_yaw_list.append(current_yaw)     # Measure yaw angle from IMU + odometry
        #self.meas_yaw_rate_list.append(0.1)  # Measure yaw rate from IMU


        self.target_yaw_list.append(target_yaw)

        cmd_yaw_rate = self.pid.update_control(target_yaw, current_yaw)
        self.cmd_yaw_rate_list.append(cmd_yaw_rate)

        msg_twist = Twist()
        msg_twist.angular.z = -cmd_yaw_rate/1000
        msg_twist.linear.x = self.speed

        self.pub_vel.publish(msg_twist)


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Command Yaw')
    parser.add_argument('--speed', type=float, default=0.3, help='Yaw angle (rad)')
    args, unknown = parser.parse_known_args() 
    follow(args.speed,5, 0, 0)

    rospy.spin()   
    print("a")


#!/usr/bin/env python

from math import sqrt
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import NavSatFix, Imu
import matplotlib.pyplot as plt
import utm

class postureMonitor:

    def __init__(self,  delta_pos: float = 2.):                      # How much to move to record
        self.start = True
        self.x = None
        self.y = None
        self.deltaPos = delta_pos
        self.prev_time = 0
        self.speed_l = []        # List for speed
        self.acc_l = []
        self.tr_acc_l = []
        self.jerk_l = []
        self.tr_jerk = []
        self.time_l = []
        self.start_t = 0
        #self.motion = 0
        #self.motion_l = []
        rospy.init_node('posture_monitor')
        sub_plot = message_filters.Subscriber('gps/fix', NavSatFix)
        sub_ori = message_filters.Subscriber('gps/imu', Imu)
        sync = message_filters.ApproximateTimeSynchronizer([sub_plot, sub_ori], 5, 1)
        sync.registerCallback(self.gps_callback)
        rospy.loginfo('Subscribing to gps/fix, gps/imu')

    def gps_callback(self, gps_fix, imu):
        x, y = utm.from_latlon(gps_fix.latitude, gps_fix.longitude)[0:2]
        time = rospy.Time(gps_fix.header.stamp.secs, gps_fix.header.stamp.nsecs).to_sec()
        if self.start:
            self.prev_time = time
            self.start = False
            self.x = x
            self.y = y
            self.start_t = time
        dt = time - self.prev_time
        dist = np.sqrt( (x-self.x)**2 + (y-self.y)**2)
        doPlot = dt > 1 or dist > self.deltaPos
        if doPlot:
            speed = dist / dt
            acc = imu.linear_acceleration.x
            tr_acc = imu.linear_acceleration.y
            if self.time_l:
                jerk = (acc - self.acc_l[-1])/dt
                tr_jerk = (tr_acc - self.tr_acc_l[-1])/dt
                #if abs(self.jerk_l[-1]+jerk) < abs(self.jerk_l[-1]):
                #    self.motion += 1
                #motionF = self.motion/(time - self.start_t)
            else:
                jerk = 0
                tr_jerk = 0
                # motionF = 0
            self.speed_l.append(speed)
            self.acc_l.append(acc)
            self.tr_acc_l.append(tr_acc)
            self.jerk_l.append(jerk)
            self.tr_jerk.append(tr_jerk)
            self.time_l.append(time - self.start_t)
            # self.motion_l.append(motionF)
            rospy.loginfo("Time: " + str(time -  self.start_t) + " s")
            rospy.loginfo("Speed: " + str(speed) + " m/s")
            rospy.loginfo("Longitudinal Acceleration: " + str(acc) + " m/s^2")
            rospy.loginfo("Longitudinal Jerk: " + str(jerk) + " m/s^3")
            rospy.loginfo("Centripetal Acceleration: " + str(tr_acc) + " m/s^2")
            rospy.loginfo("Centripetal Jerk: " + str(tr_jerk) + " m/s^3")
            #rospy.loginfo("Motion Frequency: " + str(self.motion) + " Hz")
            # if -0.147 < acc < 0.147 and -0.9 < jerk < 0.9 and -4<tr_acc<4 and -0.9 < tr_jerk< 0.9 and motionF > 0.3 and motionF < 0.15:
            if -0.147 < acc < 0.147 and -0.9 < jerk < 0.9 and -4<tr_acc<4 and -0.9 < tr_jerk< 0.9:
                rospy.loginfo("Posture stable!")
            else:
                rospy.loginfo("Posture Unstable!")
            rospy.loginfo("\n")
            self.prev_time = time
            self.x = x
            self.y = y
            p_s = plt.subplot(231)
            p_s.set_title("Speed")
            plt.ylabel("Speed (m/s)",fontsize=11,fontweight='bold')
            plt.xlabel("Time (s)",fontsize=11,fontweight='bold')
            plt.plot(self.time_l, self.speed_l)
            p_a = plt.subplot(232)
            plt.plot([0, self.time_l[-1]], [1.47, 1.47],color="red")
            plt.plot([0, self.time_l[-1]], [-1.47, -1.47],color="red")
            plt.text(0, 1.47, '1.47 m/s^2')
            plt.text(0, -1.47, '-1.47 m/s^2')
            plt.plot(self.time_l, self.acc_l)
            p_a.set_title("Longitudinal Acceleration")
            plt.ylabel("Longitudinal Acceleration (m/s^2)",fontsize=11,fontweight='bold')
            plt.xlabel("Time (s)",fontsize=11,fontweight='bold')
            p_j = plt.subplot(233)
            p_j.set_title("Longitudinal Jerk")
            plt.ylabel("Longitudinal Jerk (m/s^3)",fontsize=11,fontweight='bold')
            plt.xlabel("Time (s)",fontsize=11,fontweight='bold')
            plt.plot([0, self.time_l[-1]], [0.9, 0.9],color="red")
            plt.plot([0, self.time_l[-1]], [-0.9 , -0.9],color="red")
            plt.text(0, 0.9, '0.9 m/s^3')
            plt.text(0, -0.9, '-0.9 m/s^3')
            plt.plot(self.time_l, self.jerk_l)
            p_ta = plt.subplot(234)
            p_ta.set_title("Lateral Acceleration")
            plt.ylabel("Lateral Acceleration (m/s^2)",fontsize=11,fontweight='bold')
            plt.xlabel("Time (s)",fontsize=11,fontweight='bold')
            plt.plot([0, self.time_l[-1]], [4, 4],color="red")
            plt.text(0, 4, '4 m/s^2')
            plt.text(0, -4, '-4 m/s^2')
            plt.plot([0, self.time_l[-1]], [-4, -4],color="red")
            plt.plot(self.time_l, self.tr_acc_l)
            p_tj = plt.subplot(235)
            p_tj.set_title("Lateral Jerk")
            plt.ylabel("Lateral Jerk (m/s^3)",fontsize=11,fontweight='bold')
            plt.xlabel("Time (s)",fontsize=11,fontweight='bold')
            plt.plot([0, self.time_l[-1]], [0.9, 0.9],color="red")
            plt.plot([0, self.time_l[-1]], [-0.9, -0.9],color="red")
            plt.text(0, 0.9, '0.9 m/s^3')
            plt.text(0, -0.9, '-0.9 m/s^3')
            plt.plot(self.time_l, self.tr_jerk)
            # p_tj = plt.subplot(236)
            # p_tj.set_title("Motion Frequency")
            # plt.ylabel("Motion Frequency (Hz)",fontsize=11,fontweight='bold')
            # plt.xlabel("Time (s)",fontsize=11,fontweight='bold')
            # plt.plot([5, self.time_l[-1]], [0.1, 0.1],color="red")
            # plt.plot([5, self.time_l[-1]], [0.3, 0.3],color="red")
            # plt.text(5, 0.1, '0.1 Hz')
            # plt.text(5, 0.3, '0.3 Hz')
            # plt.plot(self.time_l, self.motion_l)
            plt.subplots_adjust(wspace =0.5, hspace =0.5)
            plt.gcf().canvas.flush_events()
            plt.show(block=False)
            plt.show(block=False)
            plt.pause(0.01)
            plt.clf()


if __name__ == '__main__':
    pm = postureMonitor()
    rospy.spin()

#!/usr/bin/env python
''' bus_monitor.py

    Monitor Autonomous Bus motion

    Task: Extend this bus monitor so that so that the vehicle heading is indicated with a black
          edge whose length is proportional to the speed.
    
    Hint 1: Add a second subscriber that reads the gps/imu topic
    Hint 2: You will need to use locks

    Copyright: Daniel Morris, 2020
'''
import argparse
from threading import Lock
import numpy as np
import rospy
import PyKDL
from sensor_msgs.msg import NavSatFix, Imu
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import utm
import math



class Color:
    ''' Obtains color for a scalar value in range minmax
        minmax: specify the range over which the colormap will span
        cmap: can optioncally provide a colormap
    '''
    def __init__(self,  minmax: Tuple[float,float], cmap=cm.get_cmap('jet',128)) -> None:
        self.cmap = cmap
        self.minmax = minmax
        self.norm = colors.Normalize(vmin=minmax[0], vmax=minmax[1])
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_color(self, val: float) -> Tuple[float,float,float,float]:
        ''' Return color for val '''
        return self.cmap( (val-self.minmax[0]) / (self.minmax[1]-self.minmax[0]) )

class PlotPosition:
    ''' Reads and plots GPS '''

    def __init__(self,  delta_pos: float = 2.,                      # How much to move before plotting another dot
                        origin_latlong: Tuple[float,float]=None,    # Optionally provide origin lat, lon (otherwise uses first measurement as origin)
                        minmax: Tuple[float,float]=(0.,8.) ):       # Range of variable for colormap

        if origin_latlong is None:
            self.origin = None
        else:
            self.origin = utm.from_latlon( *origin_latlong )[0:2]  # Initialize origin
        self.deltaPos = delta_pos
        self.minmax = minmax
        self.x = None
        self.y = None
        self.angle = 0

        rospy.init_node('bus_monitor')     

        rospy.Subscriber('gps/fix', NavSatFix,self.gps_callback)
        rospy.Subscriber('/gps/imu', Imu,self.call)
        self.my_object_lock = Lock()  
       

        rospy.loginfo('Subscribing to gps/fix, gps/imu')

    def gps_callback(self, gps_fix ):

        x,y = utm.from_latlon( gps_fix.latitude, gps_fix.longitude )[0:2]
        time = rospy.Time(gps_fix.header.stamp.secs, gps_fix.header.stamp.nsecs).to_sec()
        if self.origin is None:
            self.color = Color(minmax=self.minmax)
            self.origin = (x,y)
            self.x = x
            self.y = y
            self.prev_time=time            
            figsel = 'Bus Monitor'
            fig = plt.figure(figsel,figsize=(10,6))
            fig.clf()
            plt.subplots(1,1,num=figsel)    
            plt.grid(True, linestyle='--')
            plt.gca().set_aspect('equal', 'box')
            plt.colorbar(self.color.scalarMap, ax=plt.gca(), label='linear_speed')
            plt.gcf().canvas.flush_events()
            plt.show(block=False)
            plt.show(block=False) # For some reason calling this twice is necessary
            return

        dt = time - self.prev_time
        dist = np.sqrt( (x-self.x)**2 + (y-self.y)**2 )
        doPlot = dt > 1 or dist > self.deltaPos

        if doPlot:
            speed = dist / dt
            c = self.color.get_color(speed)  # Use linear speed for color 
            xp, yp = x-self.origin[0], y-self.origin[1]
            plt.scatter([xp], [yp], c=c)

            x1 = xp+speed/4*math.cos(self.angle)
            y1 = yp+speed/4*math.sin(self.angle)
            y1n = (x1-xp)*math.cos(math.pi*3/2) - (y1-yp)*math.sin(math.pi*3/2)+xp
            y2n =  (x1-xp)*math.sin(math.pi*3/2) - (y1-yp)*math.cos(math.pi*3/2)+yp
            plt.plot([xp, y1n],[yp, y2n],'k-')

            plt.gcf().canvas.flush_events()
            plt.show(block=False)
            plt.show(block=False) # For some reason calling this twice is necessary
            plt.pause(0.01)

            self.prev_time = time
            self.x = x
            self.y = y
    def call(self,imu):
        self.my_object_lock.acquire()
        Q = imu.orientation
        pp = PyKDL.Rotation.Quaternion(Q.x,Q.y,Q.z,Q.w)
        self.angle=float(pp.GetRPY()[2])

        self.my_object_lock.release()   




if __name__ == '__main__':

    PlotPosition()
    rospy.spin() 

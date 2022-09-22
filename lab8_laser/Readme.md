# Lab Assignment 8 - Lidar, Localization and Mapping
## ECE-CSE 434

# Preparation 

This and subsequent labs will make use of the [Greenline](https://gitlab.msu.edu/av/greenline) world package.  Start by cloning this into your catkin workspace:
```bash
$ cd ~/catkin_ws/src
$ git clone https://gitlab.msu.edu/av/greenline
```
The [Readme](https://gitlab.msu.edu/av/greenline/-/blob/master/Readme.md) explains how to run and use this world.

As usual, within your assigned repo folder, `<student_repo>`, create a package for this lab and push this to submit the assignment:
```bash
$ cd ~/catkin_ws/src/<student_repo>
$ catkin_create_pkg lab8_laser rospy
```

# Ex 1: SLAM 

Create an occupancy map of the `Greenline` world by following the SLAM instructions in its Readme file.  Make sure to explore the world to eliminate all the empty `unknown` cells over the textured ground region.  Include the saved `map.pgm` and `map.yaml` files in your `lab8_laser` folder.  

Next, have a look at the transform tree *before* and *after* starting `roslaunch greenline mapping.launch`.  You can do this with the following command:
```bash
$ rosrun rqt_tf_tree rqt_tf_tree
```
In a text file called `ex_1.txt`, identify the difference in the transform trees and explain how this difference impacts what the robot knows about itself.

# Ex 2: Obstacles in lidar
Lidar is a very useful sensor for navigation including finding obstacles.  In this exercise we will investigate clustering as a means to partition the nearby world into obstacles. 

Create a ROS node in a python file `obstacles.py` that partitions the laser scan into clusters and publishes the centroids of each cluster as a `PointCloud2` topic called `obstacles`.  Use the following criterion for clustering.  The Lidar rays are scanned every one-degree interval in azimuth, and each point has two neighbors, those on rays one-degree to its left one-degree to its right.  If the absolute difference in range of two neighbors is less than a threshold `T`, they will be in the same cluster, otherwise different clusters.  Apply this reasoning to each 360 degree scan to create clusters and publish the centroid of each cluster to `obstacles`.  Use `T=0.25 meters`. Note: do not publish any clusters at inifinity.  Your code should run with:
```
$ rosrun lab8_laser obstacles.py
```  
Hint: to get you started, have a look at: `AV Notes / ROS / python / centroid.py`, which subscribes to a `scan` topic and finds the centroid of all the lidar pixels.

When your code is working, drive the Turtlebot around the world and observe the centroids in RViz.  Now, a simple way to maintain track of objects and to determine if they are movers and predict where they may be in the future, is to track their centroids.  Based on the centroids you observe, list 3 problems with using centroids to represent objects and discriminate static from moving objects.  Write these three reasons on separate lines of a text file called: `ex_2.txt`.






# Lab Assignment 4 – Color Target Detection and ROS Messages
## ECE-CSE 434 - Autonomous Vehicles


# Introduction

This lab will explore using color cues for detecting objects, and will involve creating a new message type for passing information between ROS nodes.  Before doing this lab, it is important to review a number of sections of `AV Notes`.  In particular read:
* `AV Notes / ROS / ROS Messaging`.  You already did this in Week 2.
* `AV Notes / AV Topics / Colored Object Detection`. Implementation of Logistic Regression, which is explained in the lecture.
* `AV Notes / ROS / ROS with Multiple Threads and OpenCV`.  Care is needed when using OpenCV in ROS nodes.
* `AV Notes / ROS / ROS Launch and Parameters`.  ROS launch files are useful for bundling packages into a single command and so making it easier to call them.

The Scikit toolbox will be used for logistic regression, so to install it, first activate your python virtual environment and then install scikit-learn:
```bash
$ source ~/envs/work/bin/activate
(work)$ python -m pip install scikit-learn
```

# ROS Package
This lab assumes you have cloned your student repo into your catkin workspace as explained in lab 2.  Within this repo create your `lab4_color` package:
```
$ cd ~/catkin_ws/src/<student_repo>
$ catkin_create_pkg lab4_color rospy
```
All your code for this assignment should be in this folder:
````
~/catkin_ws/src/<student_repo>/lab4_color/src
````
Start by copying all the files in assignment repo folder: `lab4_color/src/` to the above folder.

# ROS Bags
The ROS bags for this lab are available in the same location as in lab 2: [https://drive.google.com/drive/folders/1Tx0CzG8srHAS2AYjCCD9ftkC6SUtlfRp?usp=sharing](https://drive.google.com/drive/folders/1Tx0CzG8srHAS2AYjCCD9ftkC6SUtlfRp?usp=sharing).  Alternatively, if you are using the VDI VM, you can access these for ECE students from:
```
\\cifs\courses\ECE\434\Shared\Bags
```
Or CSE students from:
```
\\cifs\courses\CSE\434\Shared\Bags
```
To mount this folder into your WSL instance, do the following for ECE (or replace ECE with CSE):
```bash
$ sudo mkdir /mnt/t
$ sudo mount -t drvfs '\\cifs\courses\ECE\434\Shared' /mnt/t
```
Then you can see the bag files with
```bash
$ ls /mnt/t/Bags
```

# Exercise 1: Single Target Detector

Train a color-target detector that will find and publish the centroid of a colored region observed by the Turtlebot. The rosbag `dot_wall_move.bag` (in the same folder as in lab 2) is a recording of a turtlebot observing a target dot on a wall whose centroid we wish to detect.  The final output of this assignment will be to detect and publish this centroid.

Generally one does not train on test data, so a training rosbag is provided for building a color-model of the dot.  This bag is called: `train_dot_wall.bag`, and again is in the same bag folder.  In one shell play this rosbag with:
```
$ rosbag play --loop ~/bags/train_dot_wall.bag
```
You can view it with
```
$ rqt_image_view
```
1. To train a color-based classifier we need an image that contains the target. Complete the `im_capture.py` node which will enable you to capture and image from a published rosbag.  It will save images to a folder `im_save`, so first cd to the `lab4_color` folder using `roscd`.   Then run `im_capture.py` to capture at least two images where the target is visible, as follows:
```
$ roscd lab4_color
$ rosrun lab4_color im_capture.py
```
2. You should now have two images of the target against the wall in folder `lab4_color/im_save` called `img_000.jpg`, and `img_001.jpg`.  Using the included code, `labeler.py`, create a mask for image `img_000.jpg` that selects the target pixels and excludes the rest.  
```
$ roscd lab4_color
$ python src/labeler.py im_save/img_000.jpg
```
3. Use `LogisticReg.py` to train a discriminative classifier on pixels that identifies target pixels.  Complete functions `apply` and `find_largest_target` in this file.  Then you can use it with the command: `python logist_reg.py <train_img_name> <train_mask_name> <test_img_name>`.  To train it with the image and mask you just created and to apply the resulting model `img_001.jpg`, run it as follows:
```
$ roscd lab4_color
$ python src/logist_reg.py im_save/img_000.jpg im_save/img_000_mask.png im_save/img_000.jpg
```
4. Let's incorporate the target detector into a ROS node that detects and plots a single target. To do this, complete `single_dot_detect.py`.  This should display a detected target and mark its centroid.  Your node should run with the command: `rosrun lab4_color single_dot_detect.py <image_name> <mask_name>`.  For your case it will be:
```
$ roscd lab4_color
$ rosrun lab4_color single_dot_detect.py im_save/img_000.jpg im_save/img_000_mask.png
```
Note: it is okay if it detects other objects with similar color to the wall marker.  What is important is that it correctly detects the centroid of the wall marker when that is the largest target.

# Exercise 2: Multi-Target Detector

The node developed in Exercise 1 has two key limitations.  First, only a single target is detected in each image, and in the future we will need to detect multiple targets per frame.  Second, the target detection is displayed but not published, and so no other nodes can use the detected centroid.  Exercise 2 will address these limitations.  It will detect multiple targets and publish their centroids to a ROS topic.  This will enable other ROS nodes to process the detected targets (which we will need in Lab 5).  For this exercise, run:
```
$ rosbag play --loop ~/bags/dot_three_wall.bag
```

1. We will need a new message type to store multiple detected centroids per frame.  There are two components it should have (1) a header, and (2) and array of points.  The header encodes time information and enables the message to be associated with the original image that it was generated from.   For the array of points, it is fine to use just the x, y coordinates of 3D points.  Hence create msg file in your lab4 ROS package called: `lab4_color/msg/Point32Array.msg` that contains the following text:
```
# Array of Point32 
std_msgs/Header header
geometry_msgs/Point32[] points
```
Now follow the directions in `AV Notes / ROS / Messaging.md / Defining Your Own Message Type` for adding and building this new message type as part of your lab 4 ROS project.  After you follow the instructions, make sure 2 things:  (a) That `rosmsg show` displays details of your message like this:
```
$ rosmsg show Point32Array
[lab4_color/Point32Array]:
std_msgs/Header header
  uint32 seq
  time stamp
  string frame_id
geometry_msgs/Point32[] points
  float32 x
  float32 y
  float32 z
```
and (b) that running `catkin_make` in folder `~/catkin_ws` completes successfully with no errors.  If you have errors, make sure to resolve them -- you have probably missed uncommenting or adding something to your `CMakeLists.txt` file. 

2. Complete `find_all_targets` in `logist_reg.py` to return the centroids of multiple targets in an image.

3. Complete `multi_dot_detect.py`.  This will detect target centroids and publish them to a topic named `/dots` using your new message type: `Point32Array`.  It does not need to do any plotting.  An example of creating a `PoseArray` (which is similar to your `Point32Array`) is in `AV Notes / ROS / python / transform_frames.py`.  Your node should run with the command:
```
$ rosrun lab4_color multi_dot_detect.py <image_name> <mask_name>
```
4. Create a node called `plot_multi_dot.py` that reads centroids from the `dots` topic and plots them on the corresponding image in the `raspicam_node/image/compressed` topic.  This will require using `message_filters` to subscribe to multiple topics, as explained in `AV Notes / ROS / Messaging`.  Use this to confirm that `multi_dot_detect.py` can successfully detect target centroids in the image topic published by playing the bag file.  Your plot node should run with:
```
$ rosrun lab4_color plot_multi_dot.py
```

5. It is a bit of a hassle to always pass running `multi_dot_detect.py` with `rosrun` and passing the images in as arguments.  Create a launch file called `dot_detect.launch` that will start `multi_dot_detect.py` and also start `plot_multi_dot.py`.  Note that you will need to pass in the absolute paths to the image and mask.  For this, you can use the `find` command described in the `AV Notes`.  When you are done, you can call the following to run multi-target detection and plotting of the centoids in the image:
```
$ roslaunch lab4_color dot_detect.launch
```

___
## Checklist
When you are done both exercises, commit and push your full `lab4_color` package including:
* `im_capture.py`
* image and mask used for training classifier
* `logist_reg.py` with `apply`, `find_largest_target`, `find_all_targets` completed
* `single_dot_detect.py`
* `Point32Array.msg`
* `multi_dot_detect.py`
* `plot_multi_dot.py`
* `dot_detect.launch` 


# DofBot_YOLO_Tracking_Catch

The project is to use [Dofbot(NIVIDIA Jetson Nano version)](https://category.yahboom.net/products/dofbot-jetson_nano) by Yahboom to recognize/trace/catch objects.

Instead of using the old version of YOLO in Dofbot, it works using [YOLOv5](https://github.com/ultralytics/yolov5).

I used ROS, YOLOv5, and python. but you can operate Dofbot without knowing it.

**However, you need your own dataset to track the object you want.**

So use YOLOv5 to make your own dataset and put it in Dofbot.

![dfdf](https://user-images.githubusercontent.com/117415885/205278535-0fe3520a-3fbb-46e5-90b2-b97a8fb834fb.png)

# Dofbot Environment

* OS : [Dofbot default OS](http://www.yahboom.net/study/Dofbot-Jetson_nano)

* python : 3.6.9 (default version)

* numpy : 1.18.5

* pytorch : 1.8.0 with cuda

* torchvision : 0.9.0

* opencv : 4.1.2 with cuda

*Instead of using the version of the module installed by default in the OS, install the corresponding version above on the dofbot.*

# Core file

3 files form the core. [ _**detect.py / detect_tracking_object.py / detect_catch_object.py**_ ]

* detect.py

The robot arm is fixed and recognizes and displays objects through the Dofbot's camera.

* detect_tracking_object.py

The robot arm tracks and moves the recognized object.

* detect_catch_object.py

The robot arm extends toward the recognized object and performs a grasping motion.

# How to use

1. Download all of the project files and put them in Dofbot.
2. Run the file that fits your purpose and follow the instructions below

All three files have a statement at the bottom that needs to be copied, and you can replace the path to the dataset and copy it to the terminal and type it.


* detect.py

```
python detect.py --nosave --weight runs/train/fruit_yolov5s_more_result2/weights/best.pt --source 0
```

* detect_tracking_object.py

```
python detect_tracking_object.py --nosave --weight runs/train/fruit_yolov5s_more_result2/weights/best.pt --source 0
```

* detect_catch_object.py

```
python detect_catch_object.py --nosave --weight runs/train/fruit_yolov5s_more_result2/weights/best.pt --source 0
```

Make sure to remove the " # " in front of the code and enter it in the terminal.

# Running image

For the test, I learned about 30 simple chapters. For basic tracking of YOLOv5, I recommend learning more than 1000 images.

![stst](https://user-images.githubusercontent.com/117415885/205296864-4177549f-84d1-4df6-8b78-a2b8e5621643.png)

![trtr](https://user-images.githubusercontent.com/117415885/205426970-f748aca3-269f-46e0-9bfb-9a49de2d63f1.png)

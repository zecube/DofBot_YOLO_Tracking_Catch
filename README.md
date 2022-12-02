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


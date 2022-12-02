import subprocess

subprocess.run(['python', '.\detect.py', '--weight', './runs/train/fruit_yolov5s_more_result2/weights/best.pt', '--source', '0'])
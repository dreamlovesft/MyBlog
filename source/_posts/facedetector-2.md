---
title: 人脸检测（二）
date: 2019-01-12 14:48:34
categories:
- OpenCV
tags:
- Python
- 人脸检测 
---


&#160; &#160; &#160; &#160;本篇博客接上一篇博客，是人脸检测算法的第二部分，讲述如何检测视频中的人脸，也是为了巩固一下所学到的知识。
<!-- more -->
&#160; &#160; &#160; &#160;本文中的视频有两种：其一为本地保存的视频，其二为来自于电脑摄像头的视频。有了上一篇博客作为基础，所以操作起来比较容易。所用到的的文件挂在[Guihub](https://github.com/dreamlovesft/CamFaceDetector)

# 导入类库 

    # USAGE
    # python webcam.py qq
    
    # import the necessary packages
    from pyimagesearch.facedetector import FaceDetector
    from pyimagesearch import imutils
    # from picamera.array import PiRGBArray
    # from picamera import PiCamera
    import argparse
    import time
    import cv2
    
&#160; &#160; &#160; &#160;这里的imutils包中包含用于执行基本图像操作的便捷功能，例如调整大小。
# 传参
&#160; &#160; &#160; &#160;这里仍然用解析命令行。如下：
```
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required = True,
    help = "path to where the face cascade resides")
ap.add_argument("-v", "--video",
    help = "path to the (optional) video file")
args = vars(ap.parse_args())
```

# 实例化算法
&#160; &#160; &#160; &#160;下面我们进行基于视频人脸检测算法的实例化，这里视频主要来源于本地或者电脑的摄像头。

      if not args.get('video', False):
          fd = FaceDetector(args["face"])
          time.sleep(0.1)
          camera = cv2.VideoCapture(0)
          while True:
              # Capture frame-by-frame
              ret, frame = camera.read()
              frame = imutils.resize(frame, width = 300)
      
              gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
              faceRects = fd.detect(gray, scaleFactor = 1.1, 
              minNeighbors = 5,minSize = (30, 30))
              frameClone = frame.copy()
      
      
              for (fX, fY, fW, fH) in faceRects:
                  cv2.rectangle(frameClone, (fX, fY), (fX+fW, fY+fH), (0,255,0), 2)
      
              cv2.imshow("Face", frameClone)
      
              if cv2.waitKey(1) & 0xFF==ord('q'):
                  break
      
          camera.release()
          cv2.destroyAllWindows()    
      
      else: 
          fd = FaceDetector(args["face"])
          camera = cv2.VideoCapture(args["video"])
          # keep looping
          while True:
              #grabing the next frame in the video by calling the read() method of camera
              (grabbed, frame) = camera.read()
              # if we are viewing a video and we did not grab a
              # frame, then we have reached the end of the video
              if args.get("video") and not grabbed:
                  break
      
              # resize the frame and convert it to grayscale
              frame = imutils.resize(frame, width = 300)
              gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
              # detect faces in the image and then clone the frame
              # so that we can draw on it
              faceRects = fd.detect(gray, scaleFactor = 1.1, minNeighbors = 5,
                  minSize = (30, 30))
              frameClone = frame.copy()
      
              # Loop over the face bounding boxes and draw them
              for (fX, fY, fW, fH) in faceRects:
                  cv2.rectangle(frameClone, (fX, fY), (fX+fW, fY+fH), (0,255,0), 2)
                  
              # show our detected faces
              cv2.imshow("Face", frameClone)
              # cleanup the camera and close any open windows
              camera.release()
              cv2.destroyAllWindows()

&#160; &#160; &#160; &#160;代码段里使用了一个if语句来实现判断视频是源自本地还是摄像头。其中，if not args.get('video', False):...，else:..表示如果解析命令行参数是没有--vedio参数，则打开摄像头，获取实时视频，否则读取本地视频。
效果如下：

![GIF3.gif](https://i.loli.net/2019/01/12/5c39cfbfd45d5.gif)














































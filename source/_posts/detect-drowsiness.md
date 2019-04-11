---
title: 基于人脸特征点的防瞌睡警报器
date: 2019-03-31 07:50:03
categories: 
- OpenCV
tags:
- Python
- Eye aspect ratio
---


# 前言
本文介绍一种基于人脸特征点检测的防瞌睡警报器。
<!-- more -->
本文通过检测人脸眼睛纵横比的变化来判定驾驶人是否处在瞌睡状态。一般的，当人的眼睛睁开时，其纵横比在某一定值附近上下波动，而闭上眼睛时其纵横比的值会发生断崖式的陡降，我们称其处于谷底状态。我们对视频每一帧中的眼睛纵横比进行检测，当眼睛纵横比处于谷底状态超过某一阈值（帧数），我们判定驾驶人处于瞌睡状态，从而产生警报。
# 人脸特征点
人脸关键点检测是人脸识别和分析领域中的关键一步，它是诸如自动人脸识别、表情分析、三维人脸重建及三维动画等其它人脸相关问题的前提和突破口。人脸轮廓主要包括眉毛、眼睛、鼻子、嘴巴和脸颊５个部分，有时还会包括对其他后续科研问题有重要价值的瞳孔和鼻孔位置。如图１所示，实现对人脸轮廓较为完整的描述，一般所需要的特征点数目在60个左右。
<div align="center">![1.jpg](https://i.loli.net/2019/03/31/5ca008757f90a.jpg)图1</div>  
# 人脸特征点的提取
在图像上进行人脸特征点提取，等价于寻找每个人脸轮廓特征点在人脸图像中的对应位置坐标，
即特征点定位（localize facial landmarks）。这一过程需要基于特征点对应的特征进行．只有获得了能够清晰标识特征点的图像特征，并依据此特征在图像中进行恰当搜索比对，在图像上精确定位特征点位置才能得以实现。获得一组人脸特征点后，我们要从中提取眼睛特征点区域，并将其转化为数组。
# 眼睛纵横比
眼睛纵横比即垂直眼睛特征点与水平眼睛特征点之间欧式距离比。当眼睛睁开时，眼睛纵横比的返回值将近似恒定。然后，该值将在闭眼期间快速减小到零。参见图2。
<div align="center">![2.jpg](https://i.loli.net/2019/03/31/5ca014cfd8e6f.jpg)图2</div>  
在图2左上角，我们的眼睛完全打开，眼睛的面部特征点如图。然后在右上角，我们的眼睛是闭着的，眼睛的面部特征点如图。然后图2底部绘制眼睛纵横比随时间的变化。可以看出当眼睛睁开时，眼睛纵横比在0.25附近波动，眼睛闭上时纵横比几乎跃变为0。
在本文中，我们将监测驾驶人眼睛纵横比的值，看看该值是否下降且持续一段时间，从而认为该人已闭上眼睛。
# 防瞌睡警报器的实现
下面我们将通过代码块的进行详细的说明。
## 导入必要的包
    #import necessary pachage
    from scipy.spatial import distance as dist
    from imutils.video import VideoStream
    from imutils import face_utils
    from threading import Thread
    import numpy as np
    import playsound
    import argparse
    import imutils
    import time
    import dlib
    import cv2

说明如下：SciPy软件包，可以计算眼睛纵横比计算中人脸特征点之间的欧几里德距离；imutils软件包，本文一系列计算机视觉和图像处理功能，都将使用到它；Thread类，可以实现在主线程的单独线程中播放警报，以确保我们的脚本在警报响起时不会暂停执行；playound库，实现WAV / MP3格式警报的播放。dlib库，实现检测和定位人脸特征点；
## 定义报警函数
接下来，我们需要定义我们的报警（sound_alarm）函数，该函数的参数为音频文件的路径，然后实现该文件的播放。
```
#define sound_alarm function
def sound_alarm(path):
    playsound.playsound(path)
 ```
 ## 眼睛纵横比计算函数
 我们还需要定义眼睛纵横比计算（eye_aspect_ratio）函数，该函数用于计算垂直眼睛特征点与水平眼睛特征点之间距离的比。
 
    #define the eye_aspect_ratio function
    def eye_aspect_ratio(eye):
        #compute the euclidean distances between the two sets of
        #vertical eye landmarks (x,y)-coordinates
        A = dist.euclidean(eye[1],eye[5])
        B = dist.euclidean(eye[2],eye[4])
        # compute the euclidean distances between the two sets of
        # horizontal eye landmarks(x,y)-coordinates
        C = dist.euclidean(eye[0],eye[3])
        # compute the eye aspect ratio 
        ear = (A + B)/(2.0 * C)
        # return the eye  aspect ratio
        return ear
        
当眼睛睁开时，眼睛纵横比的返回值将近似恒定。然后，该值将在闭眼期间快速减小到零。
## 命令行参数的解析
```
# parse our command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmarks predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
    help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int,default=0,
    help="index of webcam on system")
#--webcam : This integer controls the 
#index of your built-in webcam/USB camera.
args = vars(ap.parse_args())
```
* --shape-predictor : dlib预先训练好的面部特征点检测器的路径。
* --alarm : 指定要用作警报的输入音频文件的路径。
* --webcam ：此整数控制内置网络摄像头/ USB摄像头的索引。 

## 定义必要的变量

```
    # define two constants, one for the eye aspect ratio to indicate
    # blink and then a second constant for the number of consecutive
    # frames the eye must be below the threshold for to set off the
    # alarm
    eye_ar_thresh = 0.3 #adjusting the two constants to
    eye_ar_constant_frames = 15 #change the  Sensitivity of the drowsiness detector
    
    # initialize the frame counter as well as a boolean used to
    # indicate if the alarm is going off
    COUNTER = 0
    alarm_on = False
```
        
定义eye_ar_thresh，如果眼睛纵横比低于此阈值，我们将开始计算该人闭眼的帧数；如果该人闭眼的帧数超过eye_ar_constant_frames），我们会发出警报。调整这两个值，可以改变该模型的灵敏度。定义COUNTER，即眼睛纵横比低于eye_ar_thresh的连续帧的总数；如果COUNTER超过eye_ar_thresh，那么我们将更新布尔值alarm_on。
## 实例化人脸特征检测器
dlib库附带了直方图定向梯度的人脸检测器以及人脸特征点预测器，我们在以下代码块中实例化这两者：
```
print ("[INF0] loading facial landmarks predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
```
## 提取眼部区域
要从一组面部标志中提取眼部区域，我们只需要知道正确的数组切片（array slice）索引：

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively from a set of facial landmarks
    (LStart, LEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (RStart, REnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    # Using these indexes, we’ll easily be able to 
    # extract the eye regions via an array slice.
    
    
使用这些索引，我们可以轻松地通过数组提取眼部区域。

## 实例化报警器
```
# start the video stream thread
print("[INF0] start video stream thread..." )
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)
```
我们使用提供的--webcam索引实例化我们的VideoStream,然后我们暂停一秒钟以让相机传感器预热,我们开始在视频流中循环监测帧，并不断读取下一帧，然后我们将每一帧调整到宽度为450像素并将其转换为灰度进行预处理。接着，应用dlib的人脸检测器（detector）来查找和定位图像中的人脸。
## 定位人脸特征区域

       # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
        
            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
        
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            
我们在每个检测到的面上进行循环监测。一般在实现中，我们只有一个驾驶员，因此我们假设只有一个脸，但是我在这里留下了这个循环，可以实现在具有多个脸部的视频中的应用。然后，对于每个检测到的面部，我们应用dlib的人脸特征点检测器并将结果转换为NumPy数组，通过使用NumPy数组切片，我们可以分别提取左眼和右眼的（x，y）坐标。在获得双眼的（x，y）坐标之后，我们可以计算它们的眼睛纵横比，并取两只眼睛纵横比的平均数以增强算法的稳健性。
## 可视化眼部特征区域
然后我们可以使用下面的cv2.drawContours函数可视化我们帧上的每个眼部区域，这对我们尝试调试脚本并希望确保正确检测和定位眼睛很有用：
```
        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
```
## 监测驾驶员是否进入瞌睡状态
我们这一部分进行监测视频流中的人是否开始出现瞌睡症状。

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
            if ear < eye_ar_thresh:
                COUNTER += 1
    
                # if the eyes were closed for a sufficient number of
                # then sound the alarm
                if COUNTER >=eye_ar_constant_frames:
                    # if the alarm is not on, turn it on
                    if not alarm_on:
                        alarm_on = True
    
    
                        # check to see if an alarm file was supplied,
                        # and if so, start a thread to have the alarm
                        # sound played in the background
    
                        if  args["alarm"] != "":
                            t = Thread(target=sound_alarm,
                                args=(args["alarm"],))
                            t.deamon = True
                            t.start()
    
                        # draw an alarm on the frame
                        cv2.putText(frame, "DROWSINESS ALERT!!!", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    
            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
    
            else:
                COUNTER=0
                alarm_on=False
    
我们首先判断眼睛纵横比是否低于闭眼阈值（eye_ar_constant_frames），如果是，我们递增COUNTER，即该人闭眼的连续帧的总数。
如果COUNTER超过eye_ar_constant_frames，那么我们认为这个人开始打瞌睡，接着查看警报是否打开，如果没有，我们将其打开。如果脚本执行时提供了--alarm路径，则处理播放警报声。我们特别注意创建一个单独的线程，负责调用sound_alarm，以确保在声音播放完毕之前我们的主程序不会暂停。接着，为了增强警报的可视性，我们在屏幕中增加文字提醒。如果眼睛纵横比大于eye_ar_thresh，表示眼睛是开着的。如果眼睛睁开，我们重置COUNTER并确保警报关闭。
## 警报的可视化窗口的实现
```
        # draw the computed eye aspect ratio on the frame to help
        # with debugging and setting the correct eye aspect ratio
        # thresholds and frame counters
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
```

最后实现的效果如下视频。

<iframe width="560" height="315" src="https://www.youtube.com/embed/AowjIqbWirM" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>







































































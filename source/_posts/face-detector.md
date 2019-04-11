---
title: 人脸检测（一）
date: 2019-01-09 10:52:01
categories:
- OpenCV 
tags:
- Python
- 人脸检测 
---


# 前言
&#160; &#160; &#160; &#160;这是本人第一个比较像项目的小项目。嘻嘻
<!-- more -->

&#160; &#160; &#160; &#160;本次主要完成两个任务：（1）基于图片的人脸检测 （2）基于视频的人脸检测。分两篇博客讲完。

&#160; &#160; &#160; &#160;需要用到的文件夹挂在[Guihub](https://github.com/dreamlovesft/Face-Detector)包括：
* 人脸分类器（cascades），采用已训练好的 Haar cascade classifiers。此分类器被序列化为XML文件。调用cv2.CascadeClassifier将反序列化分类器，将其加载到内存中，并允许其检测图像中的脸。
* 图片文件夹（image），其中为待检测的图片。
* pyimagesearch，在其中的facedetector模块中定义了一个FaceDetector类。使用argparse来解析命令行参数。
* detect_faces， 主程序

## 导入类库
&#160; &#160; &#160; &#160;首先，我们要导入本算法所需要的包。
```
from __future__ import print_function
from pyimagesearch.facedetector import FaceDetector
import argparse
import cv2
```
&#160; &#160; &#160; &#160;使用argparse来解析命令行参数,要养成良好的习惯。
## 传参
&#160; &#160; &#160; &#160;接下来是参数接收阶段，我感觉也可以理解为有点预处理阶段的意思（也算是一种类吧），代码段长这样： 

     ap = argparse.ArgumentParser()
     ap.add_argument("-f", "--face", required=True,
     	help='path to where the face cascade resides')
     ap.add_argument("-i", "--image", required=True,
     	help='path to where the image file resides')
     args = vars(ap.parse_arg())
     
     image = cv2.imread(args["image"])
     gray = cv2.cvtColor(image, cv2.COLOER_BGR2GRAY)


&#160; &#160; &#160; &#160;就像前面说的那样，参与解析命令行参数来接受，我理解为这样做能尽量保持源码的封装性；两个参数即（-f/--face）cascades分类器所处路径及(-i/--image)待检测图片所处路径。

## 建立自定义类
&#160; &#160; &#160; &#160;我们需要定义一个类来处理我们如何在图像中找到脸。没错就是 facedetector.py这个东东 ，它里面长这样。
```
# import the necessary packages
import cv2

class FaceDetector:
	def __init__(self, faceCascadePath):
		# load the face detector
		self.faceCascade = cv2.CascadeClassifier(faceCascadePath)

	def detect(self, image, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30)):
		# detect faces in the image
		rects = self.faceCascade.detectMultiScale(image,
			scaleFactor = scaleFactor, minNeighbors = minNeighbors,
			minSize = minSize, flags = cv2.CASCADE_SCALE_IMAGE)

		# return the rectangles representing bounding
		# boxes around the faces
		return rects
```
&#160; &#160; &#160; &#160;这个类就是用来封装执行面部检测所需的所有逻辑的。说一下这里的__init__()是一个用来初始化或者说实例化的“壳”，self就是实例化后的函数自己，参数是cascades分类器所处路径。detect()是实际执行待检测图片有没有脸的，需要传递的参数是待检测图片。后面的三个常量参数的作用我搬来了，是这样子滴：
* scaleFactor：在每个图像比例下图像尺寸减少了多少。此值用于创建比例金字塔，以便检测图像中多个比例的脸（某些面可能更接近前景，因此更大;其他面可能更小，在背景中，因此使用不同尺度）。值为1.05表示J在金字塔的每个级别上将图像的大小减少了5％。
* minNeighbors: 每个窗口应该有多少个邻居，窗口中的区域被认为是一个脸。级联分类器将检测面部周围的多个窗口。此参数控制需要为要标记为面的窗口检测多少个矩形（邻居）。
* minSize: 宽度和高度（以像素为单位）的元组，用于指示窗口的最小尺寸。小于此大小的边界框将被忽略。一般操作是从（30,30）开始并从那里进行微调。

&#160; &#160; &#160; &#160;通过调用在FaceDetector类的构造函数中创建的分类器的detectMultiScale方法，检测图像中的实际面
## 实例化算法
&#160; &#160; &#160; &#160;以上算法模型构建完成，接下只需要将模型实例化即可。代码段长这样：
 
    fd = FaceDetector(args['face'])
    faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=5,
    	minSize=(30,30))
    print("I found {} face(s)".format(len(faceRects)))
    
    for (x, y, w, h) in faceRects:
    	cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
    
    cv2.imshow('face', image)

&#160; &#160; &#160; &#160;这里说一下for循环是为了在图像周围实际绘制一个边界框。同样，每个边界框只是一个有四个值的元组：x和y的起始位置在图像，然后是脸部的宽度和高度。好了第一部分至此完成了，但是要说明的是对一些图像可能其中的脸并不能识别出来，或者有些框里并没有人脸，那么只需调整一下scaleFactor的值，因为他是最敏感的。当然如果无论怎么调参人脸都检测不出来，那我们不要在浪费时间了，放弃吧。。。。
下面是算法的结果。
![f1.png](https://i.loli.net/2019/01/12/5c39ce030126f.png)


![f2.png](https://i.loli.net/2019/01/12/5c39cc7fc8223.png)

















































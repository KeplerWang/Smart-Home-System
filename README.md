# 智能家具系统

## 概述

[这是华中科技大学电子信息与通信学院2018级硬件课程设计的一个项目](https://github.com/KeplerWang/SmartHomeSystem)

基于树莓派、Django、socket、CNN、Ubuntu Server，本项目以''智能+家具+系统''的模式，实现了一个端到端的智能家具自控和人控系统。

作者：王子辰（负责人）、饶龙、侯梁博

指导老师：曾喻江

### Tips: 由于Face的模型过大无法上传至github，如果需要欢迎issues
### Tips: Due to the large size of our face model, it's hard to upload it to github. So welcome to issues if you need it.
## 家具部分

* 风扇（小风扇模拟）
* 加湿器
* 窗帘（步进电机模拟）
* 摄像头
* 门（舵机模拟）
* 报警器（蜂鸣器模拟）
* LED灯
* 小音响

## 智能部分
* 从温湿度传感器获取温湿度数据，据此自动控制继电器，从而控制风扇和加湿器的启动与停止
* 光敏传感器获取感光状态，自动控制窗帘的拉下和升起（即步进电机的顺时针和逆时针转动）
* 两个按键触发控制人脸识别/注册程序的启动，调用摄像头获取照片并使用FTP上传，通过socket与Ubuntu服务器通信，控制其载入模型并返回识别/注册结果
* 火焰传感器获取火焰情况，自动控制蜂鸣器的报警

## 系统部分

* 在client端，树莓派将家具设备互连，通过传感器实现家具的自动控制
* 在server端，搭建集成FTP、socket server、数据库、基于django实现的前后端分离的Web网站、人脸识别模型的Ubuntu服务器，为树莓派提供数据高效的数据处理能力和边缘计算性能
* 通过Ubuntu服务器收集、处理和存储数据显示在Web界面，用户据此查看家具设备的状态和信息，了解历史情况（如温湿度变化），监控人脸识别的结果，手动控制家具设备

## 项目使用指南

我们假设你已经完成以下工作：

* 拥有一块树莓派
* 配置Cuda服务器（非必需，如果你的CPU足够强大）
* 拥有上述所列的家具设备和传感器
* 在服务器安装MySQL、FTP等服务

你可以按照以下步骤（或其他对应的等效操作）完成项目的部署：

1. 安装任何缺少的Python依赖库（如NumPy、OpenCV、Pytroch、ftplib、pymysql、scikit-learn==0.23.0等），之后我们会上传requirement.txt，欢迎PR。
1. 下载本仓库的三个目录，将Client目录上传至树莓派，将Server和djangoProject上传至服务器中。我强烈建议你将目录都放在Desktop里。
1. 考虑到你可能会测试人脸识别，所以我先告诉你如何将自己注册到人脸数据库中：
    
    * 打开Server/FaceRecognition.py，将以下代码取消注释
    ```python
    200: from FaceCollection import face_collection
    201: if if_collect:
    202:     face_collection(size, 1, name, host)
    ...
    261     register_new('your name', host=True)
    ```
    这将采集你1000张照片，使用MTCNN-FaceNet-SVM将你注册到人脸数据库中。注册完成后务必将其恢复注释。

    * 取消注释259行，你可以从弹出的框中看到你的识别结果。
    ```python
    259:    camera_trial()
    ```
1. 在Server/server.py中，将'localhost'更改为服务器的IP地址。
    ```python
    115: if __name__ == '__main__':
    116:     UbuntuServer(('localhost', 8025)).serve_forever()
    ```
1. 在Server/mysql.py和django/mysql.py中，更改MySQL数据库的账号和密码，以及你新创建数据库的名字(test)。
    ```python
    16: try:
    17:     database = pymysql.connect(host='localhost', 
    19:                user='root', password='123456', 
    21:                port=3306, db='test')
    22:     cur = database.cursor()
    23: except pymysql.err.OperationalError:
    24:     database = pymysql.connect(host='localhost', 
    25:                user='root', password='123456',
    26:                port=3306)
    ```
1. 在Client/PiClient.py中，更改ftp用户名和密码
    ```python
    100&111:        self.f.login('username', 'password')
    ```
1. 在Client/main.py中，将'localhost'更改为服务器的IP地址。
    ```python
    25: ip_addr = ('localhost', 8025)
    ```
1. 按照Client/main.py的GPIO引脚的映射，连接家具设备和传感器。
    ```python
    ##########################################################
    # type                                  GPIO(BCM)        #
    # button for test_face                  21               #
    # button for register                   20               #
    # buzzer                                16               #
    # fire sensor                           12               #
    # light sensor                          5                #
    # relay for fan                         24               #
    # relay for wetter                      23               #
    # servo                                 27               #
    # stepper motor                         26, 19, 13, 6    #
    # temperature with humidity             17               # 
    ##########################################################
    ```
1. 打开服务器的terminal
    ```bash
    cd ~/Desktop/djangoProject
    python manage.py runserver 0.0.0.0:8000
    ```
    新建一个terminal
    ```bash
    cd ..
    cd Server
    python server.py
    ```
1. 打开树莓派的terminal
    ```bash
    cd ~/Desktop/Client
    python main.py
    ```
# Enjoy!!

## 文件结构树
```
.
├── 1.md
├── Client
│   ├── FaceCollection.py
│   ├── PiClient.py
│   ├── dataset
│   │   ├── person1
│   │   │   └── person1
│   │   └── person2
│   │       └── person2
│   ├── detected_save
│   │   ├── Unknown
│   │   ├── drawn
│   │   └── test
│   ├── hardware
│   │   ├── Button.py
│   │   ├── Buzzer.py
│   │   ├── FireSensor.py
│   │   ├── LightSensor.py
│   │   ├── MQ2.py
│   │   ├── Relay.py
│   │   ├── Servo.py
│   │   ├── StepperMotor.py
│   │   └── TemperatureWithHumidity.py
│   └── main.py
├── README.md
├── Server
│   ├── FaceCollection.py
│   ├── FaceDetection.py
│   ├── FaceRecognition.py
│   ├── dataset
│   │   ├── person1
│   │   │   └── person1
│   │   └── person2
│   │       └── person2
│   ├── dataset_cropped
│   │   ├── person1
│   │   │   └── person1
│   │   └── person2
│   │       └── person2
│   ├── detected_save
│   │   ├── Unknown
│   │   ├── drawn
│   │   └── test
│   ├── model
│   │   ├── InceptionResNetV1.py
│   │   ├── eval.py
│   │   ├── model_path
│   │   │   ├── casia-webface.pt
│   │   │   ├── classifier.pickle
│   │   │   ├── embeddings.pt
│   │   │   ├── id_to_name.pt
│   │   │   ├── ids.pt
│   │   │   └── threshold.pt
│   │   └── train.py
│   ├── mysql.py
│   └── server.py
└── djangoProject
    ├── APP
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-38.pyc
    │   │   ├── __init__.cpython-39.pyc
    │   │   ├── admin.cpython-38.pyc
    │   │   ├── admin.cpython-39.pyc
    │   │   ├── apps.cpython-38.pyc
    │   │   ├── apps.cpython-39.pyc
    │   │   ├── models.cpython-38.pyc
    │   │   ├── models.cpython-39.pyc
    │   │   └── views.cpython-39.pyc
    │   ├── admin.py
    │   ├── apps.py
    │   ├── migrations
    │   │   ├── 0001_initial.py
    │   │   ├── __init__.py
    │   │   └── __pycache__
    │   │       ├── 0001_initial.cpython-38.pyc
    │   │       ├── 0001_initial.cpython-39.pyc
    │   │       ├── __init__.cpython-38.pyc
    │   │       └── __init__.cpython-39.pyc
    │   ├── models.py
    │   ├── tests.py
    │   └── views.py
    ├── db.sqlite3
    ├── djangoProject
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-38.pyc
    │   │   ├── __init__.cpython-39.pyc
    │   │   ├── settings.cpython-38.pyc
    │   │   ├── settings.cpython-39.pyc
    │   │   ├── urls.cpython-38.pyc
    │   │   ├── urls.cpython-39.pyc
    │   │   ├── view.cpython-38.pyc
    │   │   ├── views.cpython-38.pyc
    │   │   ├── wsgi.cpython-38.pyc
    │   │   └── wsgi.cpython-39.pyc
    │   ├── asgi.py
    │   ├── settings.py
    │   ├── urls.py
    │   └── wsgi.py
    ├── identifier.sqlite
    ├── manage.py
    ├── mysql.py
    ├── static
    │   ├── bootstrap
    │   │   ├── css
    │   │   │   ├── bootstrap-theme.css
    │   │   │   ├── bootstrap.min.css
    │   │   │   ├── bootstrap.min.css.map
    │   │   │   └── weather.css
    │   │   ├── fonts
    │   │   │   ├── glyphicons-halflings-regular.eot
    │   │   │   ├── glyphicons-halflings-regular.svg
    │   │   │   ├── glyphicons-halflings-regular.ttf
    │   │   │   ├── glyphicons-halflings-regular.woff
    │   │   │   └── glyphicons-halflings-regular.woff2
    │   │   └── js
    │   │       ├── bootstrap.min.js
    │   │       ├── jquery.min.js
    │   │       └── matrix.js
    │   ├── bootstrap-3.3.7
    │   │   ├── css
    │   │   │   └── bootstrap.min.css
    │   │   ├── fonts
    │   │   │   ├── glyphicons-halflings-regular.eot
    │   │   │   ├── glyphicons-halflings-regular.svg
    │   │   │   ├── glyphicons-halflings-regular.ttf
    │   │   │   ├── glyphicons-halflings-regular.woff
    │   │   │   └── glyphicons-halflings-regular.woff2
    │   │   └── js
    │   │       └── bootstrap.min.js
    │   ├── dashboard.css
    │   ├── face
    │   │   └── readme.text
    │   ├── fontawesome
    │   │   ├── css
    │   │   │   └── font-awesome.min.css
    │   │   └── fonts
    │   │       ├── FontAwesome.otf
    │   │       ├── fontawesome-webfont.eot
    │   │       ├── fontawesome-webfont.svg
    │   │       ├── fontawesome-webfont.ttf
    │   │       ├── fontawesome-webfont.woff
    │   │       └── fontawesome-webfont.woff2
    │   ├── img
    │   │   ├── EIC.jpg
    │   │   ├── Wetter.png
    │   │   ├── background1.jpeg
    │   │   ├── closeCurtain.png
    │   │   ├── closeDoor.png
    │   │   ├── closeFan.png
    │   │   ├── closeLED.png
    │   │   ├── control.png
    │   │   ├── curtain.jpg
    │   │   ├── curtain.png
    │   │   ├── door.png
    │   │   ├── face.png
    │   │   ├── fan.png
    │   │   ├── favicon.ico
    │   │   ├── home.png
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   ├── led.png
    │   │   ├── login.png
    │   │   ├── look.png
    │   │   ├── openCurtain.png
    │   │   ├── openDoor.png
    │   │   ├── openFan.png
    │   │   ├── openLED.png
    │   │   ├── raspi.jpg
    │   │   ├── sci.jpg
    │   │   └── yinyang.svg
    │   ├── jquery-3.3.1.min.js
    │   ├── my-style.css
    │   └── web_5
    │       ├── 0.jpg
    │       └── 1.jpg
    └── templates
        ├── base.html
        ├── chart.html
        ├── face.html
        ├── index.html
        ├── login.html
        └── look.html

51 directories, 133 files
```
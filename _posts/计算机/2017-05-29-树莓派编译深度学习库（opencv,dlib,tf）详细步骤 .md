---
layout: post
title: 树莓派编译深度学习库（opencv,dlib,tf）详细步骤 
category: 计算机
tags: 嵌入式
keywords:  树莓派 dlib opencv boost 
description: 
---




## 预备工作     
1.  (最好自带翻墙梯子) SSH/更新源：      
   如果没有显示器，可以使用ssh来登陆管理树莓派，最新的树莓派默认关闭ssh服务（摄像头，vnc...），请在终端键入： sudo raspi-config 之后的选择及可以试了，相信我，很简单的。你需要改变更新源，设置为阿里源吧。           

打开编辑 /etc/apt/sources.list 文件，删除里面的内容，增加如下：   
```python
deb http://mirrors.tuna.tsinghua.edu.cn/raspbian/raspbian/ jessie main non-free contrib
deb-src http://mirrors.tuna.tsinghua.edu.cn/raspbian/raspbian/ jessie main non-free contrib
```     
之后请在**终端**执行：  
```python
sudo apt-get update && apt-get upgrade -y       #更新系统软件 并 更新已安装的包
```

## 编译opencv          
1. 下载安装相关包
```shell    
# cmake下载安装
sudo apt-get install build-essential cmake pkg-config     
# image IO相关：
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
# 视频相关：
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev
# GTK 窗口支持：
sudo apt-get install libgtk2.0-dev(废弃)
sudo apt-get install libgtk-3-dev
# python支持
sudo apt-get install python3-dev
# Atlas库：
sudo apt-get install libatlas-base-dev gfortran
# git clone opencv源代码
git clone https://github.com/opencv/opencv.git
```     

###  开始编译
```shell
cd opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=ON \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D BUILD_EXAMPLES=ON ..
make -j4
sudo make install  
```      
## 编译boost库
```shell
# 获取boost源代码    

wget http://101.96.10.75/ncu.dl.sourceforge.net/project/boost/boost/1.62.0/boost_1_62_0.tar.bz2      
# 解压源代码
tar xf boost_1_62_0.tar.bz2  
cd boost_1_62_0/

# 安装依赖包:
sudo apt-get install python-dev
sudo apt-get install -y build-essential
 
# 解压安装boost:
tar xfv boost_1_62_0.tar.bz2 
cd boost_1_62_0/
./bootstrap.sh 
echo $? # 0

 
# 编译:
./b2 #耗时144分钟
echo $? #0
 
# 安装:
sudo ./b2  install #耗时4分钟
echo $? #0
 
# 删除临时文件:
cd ..
rm -rf boost_1_62_0
 
 
# 添加到系统,Boost安装默认在此:/usr/local/lib/
echo '/usr/local/lib/' >> /etc/ld.so.conf
sudo ldconfig -v

```     
## 编译dlib
1. 这还有什么好说的呢？ 
```shell
# 拉取相关代码
git clone https://github.com/davisking/dlib.git
```
2. 开始编译
```shell
sudo python setup.py build
sudo python setup.py install
```


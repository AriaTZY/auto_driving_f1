# 使用深度学习实现游戏简单自动驾驶
Auto driving Formula 1 2019 Game / Go Kart with Deep Learning
## 环境要求
* anaconda3
* pytorch:1.2.0 (应该>1.0.0都可以)
* python:3.6/3.7
* tensorboard/tensorflow/tensorboardX:1.14.0   1.14.0   2.0
* easyDict:1.9
* opencv-python:3.4.2.16
------
  以下是在windows上玩需要具备的包，主要是对于win32的操作，如按键获取，虚拟按键等
> * pywin32:227
> * pypiwin32:223

## 硬件配置
  在PC windows上的硬件配置如下，游戏实时帧率约为20fps（使用cuda），获取保存帧率可以达到70-100帧，但有帧率限幅，限制在10帧左右
* 显卡：Nvidia RTX 2080 Super
* CPU：AMD 锐龙 3600X
  训练时间非常快，大约20-30min可以训练10k次迭代
  
## 使用方法（当前只支持GO Kart游戏）
1. 在windows PC上，使用左右双应用显示，将游戏页面调整到最顶端；
2. 游戏网页在左，PyCharm在右。使用Internet Explorer打开
3. 直接打开AI_play文件，在config的 MODEL_DIR_PATH 属性中更改至当前的模型路径（一般不用更改）
4. 直接run AI_play这个文件即可

游戏连接：[3D卡丁车竞速]http://www.4399.com/flash/122786_1.htm


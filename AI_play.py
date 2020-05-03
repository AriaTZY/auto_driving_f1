import pyautogui
import cv2 as cv
import numpy as np
import time
import os
from AlexNet import AlexDirection, AlexSpeed
import torch
from config import cfg, device_index
from main import mouse_callback, Num_detection, visualize_data
from press_key import UP, DOWN, LEFT, RIGHT, PressKey, ReleaseKey
if not device_index == 1:
    from get_key import key_check, stop_check, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP, key_check_hot
    from grab_screen import grab_screen


class SpeedPredictor:
    def __init__(self):
        model_path = cfg.TRAIN.MODEL_SPEED_PATH
        self.use_cuda = cfg.TRAIN.USE_CUDA

        # 建立模型
        self.model = AlexSpeed()

        assert os.path.exists(model_path), 'No Model File is Found, Please Recheck!'
        print('检测到预训练网络，正在加载....')
        load_dict = torch.load(model_path)
        self.model.load_state_dict(load_dict['model'])
        print('speed:预训练网络参数：total epochs:{}, total iters:{}'.format(load_dict['epoch'], load_dict['iters']))

        self.model.eval()

        if self.use_cuda:
            self.model.cuda()

    # 这里给如的图片是正常彩图格式的就好，图片格式要求是RGB
    def predict_speed(self, img):
        img = self.preprocess_image(img)
        img = torch.from_numpy(img).float()
        if self.use_cuda:
            img = img.cuda()

        # 模型预测
        predict = self.model(img).cpu().detach().numpy()[0][0]
        return predict

    def preprocess_image(self, img):
        img = cv.resize(img, (150, 150))  # 完全按照训练时的图像分辨率来，做两次resize
        img = cv.resize(img, (227, 227))
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        img = img[None, None, :, :]  # 增加一维度为1表示通道数 和 一个维度表示batch
        return img


class DirectionPredictor:
    def __init__(self):
        model_path = cfg.TRAIN.MODEL_DIR_PATH
        self.use_cuda = cfg.TRAIN.USE_CUDA

        # 建立模型
        self.model = AlexDirection()

        assert os.path.exists(model_path), 'No Model File is Found, Please Recheck!'
        print('检测到预训练网络，正在加载....')
        load_dict = torch.load(model_path)
        self.model.load_state_dict(load_dict['model'])
        print('direction:预训练网络参数：total epochs:{}, total iters:{}'.format(load_dict['epoch'], load_dict['iters']))

        self.model.eval()

        if self.use_cuda:
            self.model.cuda()

    # 这里给如的图片是正常彩图格式的就好，图片格式要求是RGB
    def predict_dir(self, img):
        img = self.preprocess_image(img)
        img = torch.from_numpy(img).float()
        if self.use_cuda:
            img = img.cuda()

        # 模型预测
        predict = self.model(img).cpu().detach().numpy()[0]
        predict_max = np.argmax(predict)
        print('\n网络输出结果:', predict, ' Soft-max 结果:', predict_max)
        return predict_max

    def preprocess_image(self, img):
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        # F1游戏注释下面这一段
        # img = cv.resize(img, (150, 150))  # 完全按照训练时的图像分辨率来，做两次resize
        # cv.rectangle(img, (11, 102), (144, 148), 0, -1)  # 在这里做遮挡，遮挡住轮子转动信息
        img = cv.resize(img, (227, 227))

        img = img[None, None, :, :]  # 增加一维度为1表示通道数 和 一个维度表示batch
        return img


# 按键按下持续时间，经验证持续时间为0表现最好，时间过长会造成帧率下降降低控制效果
press_time = 0.00


# 加速油门函数
def throttle():
    ReleaseKey(DOWN)
    PressKey(UP)


# 加速油门函数
def car_break():
    ReleaseKey(UP)
    PressKey(DOWN)


def turn_left():
    ReleaseKey(RIGHT)
    PressKey(LEFT)
    time.sleep(press_time)


def turn_right():
    ReleaseKey(LEFT)
    PressKey(RIGHT)
    time.sleep(press_time)


def go_straight():
    ReleaseKey(LEFT)
    ReleaseKey(RIGHT)
    time.sleep(press_time)


def ReleaseAllKey():
    ReleaseKey(UP)
    ReleaseKey(DOWN)
    ReleaseKey(LEFT)
    ReleaseKey(RIGHT)


if __name__ == '__main__':
    save_time = time.time()
    start_time = save_time

    # 根据机型的不同，对应不同的速度切割窗口（稍有不同）
    if device_index == 0:  # PC台式
        crop = [666, 786, 744, 784]  # 格式是 [y1, y2, x1, x2]
    elif device_index == 2:  # 笔记本
        crop = [680, 700, 745, 785]
    else:
        print('Linux Couldn`t Play F1 Game')
        crop = []
        raise Exception

    # 模式选择，选择是控制速度还是方向，对于卡丁车来说，就填方向即可
    predict_mode = 'dir'  # 'speed'

    # 读入速度检测模型
    num_detection = Num_detection()
    # 读入速度预测器
    if predict_mode == 'speed':
        predictor = SpeedPredictor()
    else:
        predictor = DirectionPredictor()

    start_time = time.time()
    while True:
        # 控制暂停与否
        if stop_check():
            print('暂停')
            ReleaseAllKey()
            time.sleep(0.5)
            flag = True
            while flag:
                flag = not stop_check()
                if not flag:
                    time.sleep(0.5)
            print('继续')

        # 图像截取部分
        # img = grab_screen(region=[0, 50, 1020, 760])  # F1屏幕抓取范围
        img = grab_screen(region=[27, 185, 666, 666])
        img = np.asarray(img)
        show_img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        # 其他部分
        key = key_check_hot()
        control = key

        real_speed = 0

        print('网络处理时间:{:.3}s'.format(time.time() - start_time))
        start_time = time.time()

        # 使用Ai预测当前速度
        control_info = [0, 0, 1, 0, 150]
        if predict_mode == 'speed':
            predict_speed = predictor.predict_speed(img)
            print('time interval:', time.time()-start_time)
            start_time = time.time()
            control_info[4] = predict_speed*320
            # 控制车辆速度
            if real_speed < predict_speed * 300:
                throttle()
            if real_speed > predict_speed * 300:
                car_break()

        elif predict_mode == 'dir':
            predict_dir = predictor.predict_dir(img)
            # 当场控制
            if predict_dir == 0:
                control_info[:2] = [0, 0]
                print('Operation: Go Straight')
                go_straight()
            elif predict_dir == 1:
                control_info[:2] = [1, 0]
                print('Operation: Turn Left')
                turn_left()
            elif predict_dir == 2:
                control_info[:2] = [0, 1]
                print('Operation: Turn Right')
                turn_right()

        # 卡丁车专用，即一直踩油门。如果是F1注释这一段
        predict_speed = 50
        if real_speed < predict_speed:
            throttle()
        if real_speed > predict_speed:
            car_break()

        # 显示速度表
        show_img = visualize_data(show_img, control_info)
        cv.imshow('SCREEN', show_img)
        cv.waitKey(1)


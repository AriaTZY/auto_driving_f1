import pyautogui
import cv2 as cv
import numpy as np
import time
import os
from get_key import key_check, char_check, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP, key_check_hot
from grab_screen import grab_screen


def line_detection(img, scope_threshold=0.2):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = gray[360:]

    edge = cv.Canny(gray, 50, 150, apertureSize=3)  # apertureSize是sobel算子大小，只能为1,3,5，7

    # threshold：累加平面的阈值参数，int类型，超过设定阈值才被检测出线段，值越大，基本上意味着检出的线段越长，检出的线段个数越少。
    # min_theta：线断检出的最小长度
    # max_theta：判断为一条线的最大允许线段间隔
    lines = cv.HoughLinesP(edge, 1.0, np.pi/180, threshold=150, minLineLength=100, maxLineGap=60)

    # 计算斜率并且筛选
    lines = lines.reshape([-1, 4])
    scope = (lines[:, 3] - lines[:, 1])/(lines[:, 2] - lines[:, 0])
    keep = np.where((scope_threshold < np.abs(scope)) & (np.abs(scope) < 1.5))[0]
    lines = lines[keep]
    # print('new frame')
    for line in lines:
        x1, y1, x2, y2 = line
        cv.line(img, (x1, y1+360), (x2, y2+360), (0, 255, 255), 13)  # 开始划线

    return img


def mouse_callback(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:
        print('axis:', x, y)


class Num_detection():
    def __init__(self):
        template = []
        for i in range(11):
            file_name = 'data/speed_num/template/' + str(i-1) + '.jpg'
            template.append(cv.imread(file_name, 0))
        self.template = np.array(np.stack(template, axis=0), np.float)

    def single_detection(self, img):
        test_data = np.array(img, np.float)

        ret = abs(self.template - test_data)
        ret = np.array(np.sum(np.sum(ret, axis=1), axis=1))
        return np.argmin(ret, 0)-1

    def speed_detect(self, num_1, num_2, num_3):
        num1 = self.single_detection(num_1)
        num2 = self.single_detection(num_2)
        num3 = self.single_detection(num_3)
        # 计算实际速度
        if num3 == -1:  # 目前只有两位数或一位数
            if num2 == -1:
                speed = num1
            else:
                speed = num1 * 10 + num2
        else:
            if num1 > 3: num1 = 3  # 逻辑防止检错，因为很容易把最高位的3检测成5
            speed = num1 * 100 + num2 * 10 + num3
        return int(speed)


def visualize_data(img, control):
    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    img = cv.cvtColor(cv.resize(img, (600, 600)), cv.COLOR_RGB2BGR)
    dir = control[:4]
    speed = control[-1]/320.
    print('speed:', speed)
    # 根据control绘制前后左右的灯
    cv.circle(img, (300, 30), 20, (10, 10, 10), thickness=3)  # 前后
    cv.circle(img, (300, 100), 20, (10, 10, 10), thickness=3)
    cv.circle(img, (230, 100), 20, (10, 10, 10), thickness=3)  # 左右
    cv.circle(img, (370, 100), 20, (10, 10, 10), thickness=3)
    # 根据control绘制前后左右的灯
    if dir[2]: cv.circle(img, (300, 30), 18, (0, 255, 0), thickness=-1)  # 前后
    if dir[3]: cv.circle(img, (300, 100), 18, (0, 255, 0), thickness=-1)
    if dir[0]: cv.circle(img, (230, 100), 18, (255, 0, 0), thickness=-1)  # 左右
    if dir[1]: cv.circle(img, (370, 100), 18, (255, 0, 0), thickness=-1)
    # 绘制速度
    cv.circle(img, (300, 600), 200, (200, 0, 255), thickness=10)
    R = 150
    theta = speed * np.pi
    pointer_x = int(300 - R*np.cos(theta))
    pointer_y = int(600 - R*np.sin(theta))
    cv.line(img, (pointer_x, pointer_y), (300, 600), (255, 255, 255), thickness=5)
    cv.circle(img, (300, 600), 20, (200, 200, 255), thickness=-1)

    return img


# 读入存储的numpy数据
def read_numpy(idx):
    dir = os.listdir('data/image/')
    num_image = len(dir)
    file_name = 'data/npy/control_dataset.npy'
    control_buffer = np.load(file_name)
    assert num_image == control_buffer.shape[0], 'Shape is not equal'
    print('数据个数:', num_image)
    for i in range(idx, num_image):
        image = cv.imread(os.path.join('data/image/', str(i) + '.jpg'))
        print(os.path.join('data/image/', str(i) + '.jpg'))
        img = visualize_data(image, control_buffer[i])
        cv.imshow('window', img)
        cv.waitKey(0)


# 把缓存中的数据使用图片方式保存（这样会比直接保存numpy数组很省空间）
def save_jpeg(image_buffer, start_index=0):
    num_image = image_buffer.shape[0]
    for i in range(num_image):
        file_name = 'data/image/'+str(start_index+i)+'.jpg'
        image = image_buffer[i]
        cv.imwrite(file_name, image)
        if i % 1000 == 0:
            print('{} done 保存中....'.format(i))
    print('保存JPEG图片完成')


if __name__ == '__main__':
    num_detection = Num_detection()
    save_time = time.time()
    start_time = save_time

    numpy_path = 'data/npy/'

    image_cache = []
    control_cache = []
    read_numpy(5000)

    # 倒计时
    for i in range(3):
        time.sleep(1)
        print('With Count down of', 3-i)
    while True:
        # 控制暂停与否
        control_key = char_check()
        if len(control_key) and control_key[0] == ord('T'):
            print('暂停')
            time.sleep(1)
            flag = True
            while flag:
                check = char_check()
                if len(check) and check[0] == ord('T'):
                    flag = False
                    time.sleep(1)
            print('继续')

        # 图像截取部分
        # img = pyautogui.screenshot(region=[0, 50, 1020, 760])  # 0.07左右
        # speed_img = pyautogui.screenshot(region=[745, 730, 40, 20])  # 0.07左右
        img = grab_screen(region=[0, 50, 1020, 760])  # 0.03左右
        img = np.asarray(img)
        show_img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        # 其他部分
        key = key_check_hot()
        control = key
        # print('time interval', time.time()-start_time)
        # start_time = time.time()

        # 有关速度获取的程序段
        if True:
            # speed_img = grab_screen(region=[745, 730, 785, 750])  # 调试使用，在不需要截取全景图使用
            speed_img = img[680:700, 745:785]  # 这个是基于截的全景图的截取，在有条件的时候使用这个方法可以更快
            speed_img = cv.cvtColor(np.asarray(speed_img), cv.COLOR_RGB2GRAY)
            scale = 10.0
            y_axis = [3, 17]  # [30, 168]
            x_axis = np.array(np.arange(60, 391, 110)/10, np.int)
            num_1 = speed_img[y_axis[0]:y_axis[1], x_axis[0]:x_axis[1]]
            num_2 = speed_img[y_axis[0]:y_axis[1], x_axis[1]:x_axis[2]]
            num_3 = speed_img[y_axis[0]:y_axis[1], x_axis[2]:x_axis[3]]

            speed = num_detection.speed_detect(num_1, num_2, num_3)
            control.extend([speed])

            # print('Speed:', speed)
            # speed_img = cv.resize(speed_img, (0, 0), None, 10, 10)
            # cv.imshow('SPEED', speed_img)

        # 查看确定帧率之类的事情
        show_img = cv.cvtColor(show_img, cv.COLOR_BGR2GRAY)
        show_img = cv.resize(show_img, (150, 150))
        cv.imshow('SCREEN', show_img)
        cv.waitKey(1)

        img = cv.resize(img, (150, 150))
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        # 使用numpy数组保存数据，分图像数据和其他控制数据两个文件
        image_cache.append(img)
        control_cache.append(control)

        if len(image_cache) % 100 == 0:
            print(time.time()-start_time, len(image_cache))
            start_time = time.time()

        if len(image_cache) == 2000:
            start_idx = 0
            # 保存之前，查看是否有保存好的，如果有就读取
            if os.path.exists(numpy_path + 'control_dataset.npy'):  # 如果有control这个文件
                file_name = 'data/npy/control_dataset.npy'
                pre_control_buffer = np.load(file_name)
                dir = os.listdir('data/image/')
                if len(dir) == pre_control_buffer.shape[0]:
                    start_idx = len(dir)
                    print('验证完毕，从{}开始写入'.format(start_idx))
                else:
                    raise Exception('control文件与image文件个数不匹配，请重新检查')

            control_numpy_path = os.path.join(numpy_path, 'control_dataset.npy')
            save_image = np.stack(image_cache, axis=0)
            save_control = np.stack(control_cache, axis=0)
            # 如果之前有数据，做一次数据衔接
            if not start_idx == 0:
                save_control = np.concatenate([pre_control_buffer, save_control], 0)
                # save_control = np.stack([pre_control_buffer, save_control], 0)
                print('接上原有control数据，{}+{}={}'.format(
                    pre_control_buffer.shape[0], len(control_cache), save_control.shape[0]))
                del pre_control_buffer
            # 保存阶段
            print('开始保存')
            np.save(control_numpy_path, save_control)
            save_jpeg(save_image, start_idx)
            print(save_control.shape)
            print('保存完成!')
            # 释放空间
            del save_image, save_control,
            image_cache = []
            control_cache = []


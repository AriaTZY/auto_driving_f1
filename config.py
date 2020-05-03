from easydict import EasyDict
import numpy as np

__C = EasyDict()
cfg = __C

# 不同的电脑会有一些不同的配置，最主要的区别是文件路径是否相同
device_list = ['PC_Windows', 'PC_Linux', 'LapTop']
device_index = 0

__C.TRAIN = EasyDict()

__C.TRAIN.MODEL_SPEED_PATH = 'output/model/speed_predictor.pth'
__C.TRAIN.MODEL_DIR_PATH = 'output/model/dirction_predictor.pth'

# 有关训练参数
__C.TRAIN.LR = 0.0005
__C.TRAIN.BATCH_SIZE = 256
__C.TRAIN.KEEP_EPOCHS = 100  # 表示再训练多少个epochs

# 有关调试参数
__C.TRAIN.USE_TENSORBOARD = True
__C.TRAIN.DEBUG_INTERVAL = 2
__C.TRAIN.SAVE_INTERVAL = 2000
__C.TRAIN.VISUALIZE_INTERVAL = 1000  # 一般情况下，这个值要小于save interval

__C.TRAIN.USE_CUDA = True

# 设置不同电脑的不同配置
print('当前运行机型：', device_list[device_index])

# 台式机windows，使用绝对路径
if device_index == 0:
    # __C.TRAIN.DATA_ROOT_PATH = 'G:\\formula_1\\data\\'  # F1数据位置
    __C.TRAIN.DATA_ROOT_PATH = 'G:\\formula_1\\modified\\'  # 新数据集位置（卡丁车）
    __C.TRAIN.DATA_ROOT_PATH_VAL = 'G:\\formula_1\\modified\\val\\'  # 新数据集位置——验证集

# 台式机Linux，使用软连接
elif device_index == 1:
    __C.TRAIN.DATA_ROOT_PATH = 'data\\'
    __C.TRAIN.DATA_ROOT_PATH_VAL = 'data\\'

# 笔记本电脑，暂定
elif device_index == 2:
    __C.TRAIN.DATA_ROOT_PATH = 'data\\'
    __C.TRAIN.DATA_ROOT_PATH_VAL = 'data\\'


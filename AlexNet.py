import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import os
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import time
import tensorboardX

class DataSet_F1(Dataset):
    # 注意control给到文件名，image_dir给到文件夹
    def __init__(self, control_file, image_dir):
        super(DataSet_F1, self).__init__()
        num_image = len(os.listdir(image_dir))
        control_buffer = np.load(control_file)
        assert num_image == control_buffer.shape[0], 'Shape is not equal'
        self.num = num_image
        self.control = control_buffer
        self.img_dir = image_dir

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        image = cv.imread(os.path.join(self.img_dir, str(index) + '.jpg'), 0)
        image = cv.resize(image, (227, 227))
        image = image[None, :, :]  # 增加一维度为1表示通道数
        control = self.control[index][-1]/320.  # 把速度值归一化至0/1之间
        return image, control


class AlexNet(nn.Module):  # 定义网络，推荐使用Sequential，结构清晰
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = torch.nn.Sequential(  # input_size = 227*227*3
            torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)  # output_size = 27*27*96
        )
        self.conv2 = torch.nn.Sequential(  # input_size = 27*27*96
            torch.nn.Conv2d(96, 256, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)  # output_size = 13*13*256
        )
        self.conv3 = torch.nn.Sequential(  # input_size = 13*13*256
            torch.nn.Conv2d(256, 384, 3, 1, 1),
            torch.nn.ReLU(),  # output_size = 13*13*384
        )
        self.conv4 = torch.nn.Sequential(  # input_size = 13*13*384
            torch.nn.Conv2d(384, 384, 3, 1, 1),
            torch.nn.ReLU(),  # output_size = 13*13*384
        )
        self.conv5 = torch.nn.Sequential(  # input_size = 13*13*384
            torch.nn.Conv2d(384, 256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)  # output_size = 6*6*256
        )

        # 网络前向传播过程
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(9216, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4)
        )

    def forward(self, x):  # 正向传播过程
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        res = conv5_out.view(conv5_out.size(0), -1)
        out = self.dense(res)
        return out


class AlexSpeed(nn.Module):
    def __init__(self):
        super(AlexSpeed, self).__init__()
        self.conv1 = torch.nn.Sequential(  # input_size = 227*227*1
            torch.nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)  # output_size = 27*27*96
        )
        self.conv2 = torch.nn.Sequential(  # input_size = 27*27*96
            torch.nn.Conv2d(96, 256, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)  # output_size = 13*13*256
        )
        self.conv3 = torch.nn.Sequential(  # input_size = 13*13*256
            torch.nn.Conv2d(256, 384, 3, 1, 1),
            torch.nn.ReLU(),  # output_size = 13*13*384
        )
        self.conv4 = torch.nn.Sequential(  # input_size = 13*13*384
            torch.nn.Conv2d(384, 384, 3, 1, 1),
            torch.nn.ReLU(),  # output_size = 13*13*384
            torch.nn.MaxPool2d(3, 2)  # output_size = 6*6*256
        )
        self.conv5 = torch.nn.Sequential(  # input_size = 6*6*384
            torch.nn.Conv2d(384, 256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 3)  # output_size = 2*2*256
        )

        # 网络前向传播过程
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 1)
        )

        # self.normal_layer(self.conv1, 0, 0.01)
        # self.normal_layer(self.conv2, 0, 0.01)
        # self.normal_layer(self.conv3, 0, 0.01)
        # self.normal_layer(self.conv4, 0, 0.01)
        # self.normal_layer(self.conv5, 0, 0.01)
        # self.normal_layer(self.dense, 0, 0.01)

    def forward(self, x):  # 正向传播过程
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        res = conv5_out.view(conv5_out.size(0), -1)
        out = self.dense(res)
        return out

    def normal_layer(self, model, mean, std):
        list_model = list(model)
        for layer in list_model:
            layer.weight.data.normal_(mean, std)
            layer.bias.data.zero_()


def tonumpy(tensor):
    return tensor.detach().cpu().numpy()


if __name__ == '__main__':
    # 参数配置
    total_epochs = 100
    use_cuda = True
    use_tensorboard = True
    lr = 0.0003
    batch_size = 256
    model_path = 'output/model/speed_predictor.pth'
    exp_time = 2
    logger_path = 'output/run/exp' + str(exp_time) + '/'

    debug_interval = 2
    save_interval = 500
    visualize_interval = 100
    iters = 0
    start_epoch = 0

    # 读入数据
    dataset = DataSet_F1(control_file='data/npy/control_dataset.npy', image_dir='data/image/')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    # 建立模型
    model = AlexSpeed()
    # model = nn.Module()
    # 如果有网络读取网络
    if os.path.exists(model_path):
        print('检测到预训练网络，正在加载....')
        load_dict = torch.load(model_path)
        model.load_state_dict(load_dict['model'])
        start_epoch = load_dict['epoch']
        iters = load_dict['iters']

    model.train()

    if use_cuda:
        model.cuda()
    if use_tensorboard:
        print('USE Tensor boardX!')
        logger = tensorboardX.SummaryWriter(logger_path)

    # model = nn.Module()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    root_time = time.time()
    start_time = time.time()
    end_time = time.time()

    for epoch in range(start_epoch, total_epochs):
        for idx, (img, speed) in enumerate(dataloader):
            iters += 1
            # 清零操作
            model.zero_grad()
            optimizer.zero_grad()

            # 读入数据
            img = img.float()
            speed = speed.float()
            if use_cuda:
                img = img.cuda()
                speed = speed.cuda()
            img = Variable(img)
            speed = Variable(speed)

            # 模型预测
            predict = model(img)
            predict = predict.view([batch_size, ])

            # 构建loss以及下降
            loss = torch.nn.MSELoss()(speed, predict)
            loss.backward()
            optimizer.step()

            # 打印数据
            if iters % debug_interval == 0:
                end_time = time.time()
                print('epoch: {}/{}, index:{}, iter: {}, time: {:.3f}s / {:.3f}s'.format(epoch, total_epochs, idx,
                                                                                         iters, end_time - start_time,
                                                                                         time.time() - root_time))
                start_time = time.time()

                print('loss:{}'.format(loss.item()))
                if use_tensorboard:
                    logger.add_scalar('loss', loss.item(), global_step=iters)
                    logger.add_scalar('test_loss', np.random.rand(), global_step=iters)

            # 保存网络
            if iters % save_interval == 0:
                save_dict = dict()
                save_dict['model'] = model.state_dict()
                save_dict['iters'] = iters
                save_dict['epoch'] = epoch
                torch.save(save_dict, model_path)
                print('Save Model Successfully!')

            # 可视化验证
            if iters % visualize_interval == 0:
                from main import visualize_data
                speed = tonumpy(speed)
                predict = tonumpy(predict)
                sheet1 = np.zeros([1220, 1220, 3], np.uint8)
                sheet2 = np.zeros([1220, 1220, 3], np.uint8)
                crood = [[0, 0], [620, 0], [0, 620], [620, 620]]
                for i in range(4):
                    idx = np.random.randint(0, batch_size-1)
                    img_show = np.array(tonumpy(img)[idx][0], np.uint8)
                    gt_img = visualize_data(img_show, [0, 0, 1, 1, speed[idx]*320])
                    predict_img = visualize_data(img_show, [0, 1, 0, 1, predict[idx]*320])
                    # 为了整合在一张sheet上面，重新设定比例
                    gt_img = cv.resize(gt_img, (600, 600))
                    predict_img = cv.resize(predict_img, (600, 600))
                    # 整合在一张sheet上
                    if i < 2:
                        sheet1[crood[i*2][1]:crood[i*2][1]+600, crood[i*2][0]:crood[i*2][0]+600] = gt_img
                        sheet1[crood[i*2+1][1]:crood[i*2+1][1]+600, crood[i*2+1][0]:crood[i*2+1][0]+600] = predict_img
                    else:
                        i = i-2
                        sheet2[crood[i * 2][1]:crood[i * 2][1] + 600,
                        crood[i * 2][0]:crood[i * 2][0] + 600] = gt_img
                        sheet2[crood[i * 2 + 1][1]:crood[i * 2 + 1][1] + 600,
                        crood[i * 2 + 1][0]:crood[i * 2 + 1][0] + 600] = predict_img
                if use_tensorboard:
                    logger.add_image('sheet1', sheet1, global_step=iters, dataformats='HWC')
                    logger.add_image('sheet2', sheet2, global_step=iters, dataformats='HWC')
                else:
                    cv.imshow('ground_true', sheet1)
                    cv.imshow('prediction', sheet2)
                    cv.waitKey(0)

    if use_tensorboard:
        logger.close()


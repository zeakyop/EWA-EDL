# -*- encoding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn import functional as F
import warnings

warnings.filterwarnings("ignore")


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)  # m.weight.data是卷积核参数


def gaussian(d, conf, Z):  # 高斯函数
    ans = np.exp(-(d) ** 2 / (2 * (conf ** 2)))
    ans = ans / (np.sqrt(2 * np.pi) * conf * Z)
    return ans


def gaussian_Z(conf, EL):  # Z的计算
    ans = 0
    for i in EL:
        ans += np.exp(-(i) ** 2 / (2 * (conf ** 2)))
    ans = ans / (np.sqrt(2 * np.pi) * conf)
    return ans


# anger disgust joy sadness fear expect trust surprised
anger = [0, 1, 2, 2, 3, 4, 1, 3]
disgust = [1, 0, 3, 1, 2, 3, 2, 4]
joy = [2, 3, 0, 4, 3, 2, 1, 1]
sadness = [2, 1, 4, 0, 1, 2, 3, 3]
surprise = [3, 2, 3, 1, 0, 1, 4, 2]
fear = [4, 3, 2, 2, 1, 0, 3, 1]

X = []
X.append(anger)
X.append(disgust)
X.append(joy)
X.append(sadness)
X.append(surprise)
X.append(fear)

emotion = []
for i in range(6):
    z = gaussian_Z(1, X[i])
    Zem_dtr = [gaussian(d, 1, z) for d in X[i]]
    emotion.append(Zem_dtr)
emotion = np.array(emotion)


# THETA = 1.0

class TextCNN(nn.Module):
    def __init__(self, vec_dim, filter_num, sentence_max_size, label_size, kernel_list):
        """
        :param vec_dim: 词向量的维度  1x300
        :param filter_num: 每种卷积核的个数 100
        :param sentence_max_size:一个句子的包含的最大的词数量 15
        :param label_size:标签个数，全连接层输出的神经元数量=标签个数 6
        :param kernel_list:卷积核列表 3,4,5
        """
        super(TextCNN, self).__init__()
        chanel_num = 1
        # nn.ModuleList相当于一个卷积的列表，相当于一个list
        # nn.Conv2d的输入是个四维的tensor，每一位分别代表batch_size、channel、length、width
        # nn.MaxPool1d()是最大池化，此处对每一个向量取最大值，所有kernel_size为卷积操作之后的向量维度
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(chanel_num, filter_num, (kernel, vec_dim)),  # 1, 100, (3, 300)
            nn.ReLU(),
            # 经过卷积之后，得到一个维度为sentence_max_size - kernel + 1的一维向量
            nn.MaxPool2d((sentence_max_size - kernel + 1, 1))
        ) for kernel in kernel_list])

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(filter_num * len(kernel_list), label_size),
            nn.Softmax(dim=1),
        )
        # # 全连接层，因为有6个标签
        # self.fc = nn.Linear(filter_num * len(kernel_list), label_size)
        # # dropout操作，防止过拟合
        # self.dropout = nn.Dropout(0.5)
        # # 分类
        # self.sm = nn.Softmax(dim=1)  # dim=0
        # for cv in self.convs:
        self.convs.apply(gaussian_weights_init)
        self.fc.apply(gaussian_weights_init)

    def forward(self, x):
        # Conv2d的输入是个四维的tensor，每一位分别代表batch_size、channel、length、width
        in_size = x.size(0)  # x.size(0)，表示的是输入x的batch_size
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(in_size, -1)  # 设经过max pooling之后，有output_num个数，将out变成(batch_size,output_num)，-1表示自适应
        out = self.fc(out)  # nn.Linear接收的参数类型是二维的tensor(batch_size,output_num),一批有多少数据，就有多少行
        return out


class KL_Loss(nn.Module):
    def __init__(self):
        super(KL_Loss, self).__init__()

    def forward(self, pred, label):
        # A = label.numpy()
        # A = np.array(A, dtype=float)
        # for i in range(A.shape[0]):
        #     row_sum = np.sum(A[i])
        #     if row_sum == 0.0:
        #         A[i] = np.zeros(A.shape[1])
        #     else:
        #         A[i] = A[i] / row_sum
        # A = torch.FloatTensor(A)
        loss = 0.0
        # pe = torch.sum(torch.exp(pred), axis=1)
        for i in range(pred.shape[0]):
            # portion = torch.log(torch.div(torch.exp(pred[i]), pe[i]))
            portion = torch.log(pred[i])
            loss = torch.add(loss, torch.sum(label[i] * portion))
        loss = torch.div(loss, -pred.shape[0])
        return loss


class My_CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(My_CrossEntropyLoss, self).__init__()

    def forward(self, pred, label):
        loss = 0.0
        # pe = torch.sum(torch.exp(pred), axis=1)
        for i in range(pred.shape[0]):
            # portion = torch.log(torch.div(torch.exp(pred[i]), pe[i]))
            portion = torch.log(pred[i])
            loss = torch.add(loss, -portion[label[i]])
        loss = torch.div(loss, pred.shape[0])
        return loss


def compute_distribution(batch_pred):
    pre = batch_pred.detach().numpy()
    new_batch_pred = []
    for j in range(pre.shape[0]):  # 对batch_size个句子，根据预测值分配权重
        new_pre = []
        for k in range(6):
            pre_temp = [d * pre[j][k] for d in emotion[k]]
            new_pre.append(pre_temp)

        new_pre = np.array(new_pre)
        pre_sum = [0, 0, 0, 0, 0, 0, 0, 0]
        for k in range(6):
            pre_sum += new_pre[k]
        # pre_sum = theta * pre_sum + (1-theta) * pre[j]
        pre_sum = pre_sum[0:6]
        new_batch_pred.append(pre_sum)

    # print('new:')
    # print(new_batch_pred)

    new_batch_pred = torch.Tensor(new_batch_pred)
    return new_batch_pred


def compute_classification(test_loader, test_set):
    model = TextCNN(vec_dim=embedding_dim, filter_num=filter_num, sentence_max_size=sentence_max_size,
                    label_size=label_size, kernel_list=kernel_list)
    model.load_state_dict(torch.load('record4/model.pth'))
    model.eval()
    y_true = np.array([], dtype=int)
    y_pred = np.array([], dtype=int)
    Acc = 0.0
    for i, data in enumerate(test_loader):
        batch_pred = model(data[0])
        new_batch_pred = compute_distribution(batch_pred)
        batch_pred.data = new_batch_pred
        '''
        pre = batch_pred.detach().numpy()
        new_batch_pred = []
        for j in range(pre.shape[0]):  # 根据预测值分配权重
            new_pre = []
            for k in range(6):
                pre_temp = [d * pre[j][k] for d in emotion[k]]
                new_pre.append(pre_temp)

            new_pre = np.array(new_pre)
            pre_sum = [0, 0, 0, 0, 0, 0]
            for k in range(6):
                pre_sum += new_pre[k]
            new_batch_pred.append(pre_sum)
        # print('new:')
        # print(new_batch_pred)

        new_batch_pred = torch.Tensor(new_batch_pred)
        batch_pred.data = new_batch_pred
        '''
        tem_data = torch.argmax(data[1], axis=1)
        y_true = np.concatenate([y_true, tem_data.numpy()])
        y_pred = np.concatenate([y_pred, np.argmax(batch_pred.data.numpy(), axis=1)])
        Acc += np.sum(np.argmax(batch_pred.data.numpy(), axis=1) == tem_data.numpy())
    Acc /= test_set.__len__()
    F1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    result_name = open("result0.8.txt", "a")
    print("Pre:%3.6f Rec:%3.6f F1:%3.6f Acc:%3.6f" % (precision, recall, F1, Acc))
    print("%3.6f %3.6f %3.6f %3.6f" % (precision, recall, F1, Acc), file=result_name)
    result_name.close()
    # with open('record/acc.txt', 'a+') as f:
    #     f.write('Pre: ' + str(precision) + ' Rec: ' + str(recall) + " F1: " + str(F1) + ' Acc: ' + str(Acc) + '\n')


def train_textcnn_model(model, train_loader, val_loader, num_epoch, lr, lamda, train_set, val_set):
    print("begin training...")
    best_acc = 0.0
    min_loss = 10.0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # 优化器
    # loss1 = nn.CrossEntropyLoss()
    loss1 = My_CrossEntropyLoss()  # 损失函数1
    loss2 = KL_Loss()  # 损失函数2
    for epoch in range(num_epoch):  # 迭代循环
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        model.train()  # 将模型设置为训练模式
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()  # 优化器参数初始化
            batch_pred = model(data[0])  # 一个batch_size数据量的模型预测值
            # new_batch_pred = compute_distribution(batch_pred)
            # batch_pred.data = new_batch_pred
            '''
            pre = batch_pred.detach().numpy()
            new_batch_pred = []
            for j in range(pre.shape[0]):  # 根据预测值分配权重
                new_pre = []
                for k in range(6):
                    pre_temp = [d * pre[j][k] for d in emotion[k]]
                    new_pre.append(pre_temp)

                new_pre = np.array(new_pre)
                pre_sum = [0, 0, 0, 0, 0, 0]
                for k in range(6):
                    pre_sum += new_pre[k]
                print(pre_sum)
                print(pre[j])
                pre_sum = 0.1 * pre_sum + 0.9 * pre[j]
                new_batch_pred.append(pre_sum)
                print(pre_sum)
            # print('new:')
            # print(new_batch_pred)

            new_batch_pred = torch.Tensor(new_batch_pred)
            # print(new_batch_pred)
            batch_pred.data = new_batch_pred

            # print('new_grad:')
            # print(new_batch_pred)
            # print('new_F :')
            # print(new_batch_pred)
            '''
            tem_data = torch.argmax(data[1], axis=1)  # 计算一个batch_size数据真实情感分布的最大值索引
            # print(new_batch_pred)
            batch_loss = (1.0 - lamda) * loss1(batch_pred, tem_data) + lamda * loss2(batch_pred, data[1])  # 计算损失
            batch_loss.backward()  # 根据损失反向传播更新参数
            optimizer.step()
            # print(np.argmax(new_batch_pred.data.numpy(), axis=1))
            # print(tem_data.numpy())
            train_acc += np.sum(np.argmax(batch_pred.data.numpy(), axis=1) == tem_data.numpy())  # 预测准确值得数量
            train_loss += batch_loss.item()

        model.eval()  # 将模型设置为测试模式
        for i, data in enumerate(val_loader):
            batch_pred = model(data[0])
            new_batch_pred = compute_distribution(batch_pred)
            batch_pred.data = new_batch_pred
            '''
            pre = batch_pred.detach().numpy()
            new_batch_pred = []
            for j in range(pre.shape[0]):  # 根据预测值分配权重
                new_pre = []
                for k in range(6):
                    pre_temp = [d * pre[j][k] for d in emotion[k]]
                    new_pre.append(pre_temp)

                new_pre = np.array(new_pre)
                pre_sum = [0, 0, 0, 0, 0, 0]
                for k in range(6):
                    pre_sum += new_pre[k]
                new_batch_pred.append(pre_sum)
            # print('new:')
            # print(new_batch_pred)

            new_batch_pred = torch.Tensor(new_batch_pred)
            batch_pred.data = new_batch_pred
            '''
            tem_data = torch.argmax(data[1], axis=1)
            batch_loss = (1.0 - lamda) * loss1(batch_pred, tem_data) + lamda * loss2(batch_pred, data[1])
            val_acc += np.sum(np.argmax(batch_pred.data.numpy(), axis=1) == tem_data.numpy())
            val_loss += batch_loss.item()
        train_acc = train_acc / train_set.__len__()
        val_acc = val_acc / val_set.__len__()
        print('[%03d/%03d] Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f Loss: %3.6f' \
              % (epoch + 1, num_epoch, train_acc, train_loss, val_acc, val_loss))
        # train_acc = train_acc / train_set.__len__()
        # print('[%03d/%03d] Train Acc: %3.6f Loss: %3.6f' % (epoch + 1, num_epoch, train_acc, train_loss))
        # model_name = r'record/model' + str(lamda) + '.pth'
        # if 1:
        #     torch.save(model.state_dict(), model_name)
        #     # best_acc = val_acc
        #     # min_loss = val_loss
        #     best_acc = val_acc

    model_name = r'record4/model.pth'
    torch.save(model.state_dict(), model_name)
    # with open('record/acc.txt', 'a') as f:
    #     f.write('lamda:' + str(lamda) + '\t' + 'best_acc:' + str(best_acc) + '\n')
    #
    # print("lamda is %.2f" % lamda)
    # print("Best val_acc is %3.6f" % best_acc)
    # compute_classification(Test_DataLoader)
    print('Finished Training')


def est_textcnn_model(test_loader, lamda, test_set):
    model = TextCNN(vec_dim=embedding_dim, filter_num=filter_num, sentence_max_size=sentence_max_size,
                    label_size=label_size, kernel_list=kernel_list)
    model_name = r'record4/model' + '.pth'
    model.load_state_dict(torch.load(model_name))
    model.eval()  # 必备，将模型设置为训练模式
    test_acc = 0.0
    for i, data in enumerate(test_loader):
        batch_pred = model(data[0])
        new_batch_pred = compute_distribution(batch_pred)
        batch_pred.data = new_batch_pred
        '''
        pre = batch_pred.detach().numpy()
        new_batch_pred = []
        for j in range(pre.shape[0]):  # 根据预测值分配权重
            new_pre = []
            for k in range(6):
                pre_temp = [d * pre[j][k] for d in emotion[k]]
                new_pre.append(pre_temp)

            new_pre = np.array(new_pre)
            pre_sum = [0, 0, 0, 0, 0, 0]
            for k in range(6):
                pre_sum += new_pre[k]
            new_batch_pred.append(pre_sum)
        # print('new:')
        # print(new_batch_pred)

        new_batch_pred = torch.Tensor(new_batch_pred)
        batch_pred.data = new_batch_pred
        '''
        tem_data = torch.argmax(data[1], axis=1)
        test_acc += np.sum(np.argmax(batch_pred.data.numpy(), axis=1) == tem_data.numpy())
    test_acc = test_acc / test_set.__len__()
    print('Accuracy of the network on test set: %3.6f' % test_acc)


def cross_model(k):
    train_set_name = "save4/train_set" + str(k) + ".pth"
    train_label_name = "save4/train_label" + str(k) + ".pth"
    val_set_name = "save4/val_set" + str(k) + ".pth"
    val_label_name = "save4/val_label" + str(k) + ".pth"
    test_set_name = "save4/test_set" + str(k) + ".pth"
    test_label_name = "save4/test_label" + str(k) + ".pth"

    train_set = torch.load(train_set_name)  # 训练数据
    train_label = torch.load(train_label_name)  # 训练数据标签
    val_set = torch.load(val_set_name)  # 验证数据
    val_label = torch.load(val_label_name)  # 验证数据标签
    test_set = torch.load(test_set_name)  # 测试数据
    test_label = torch.load(test_label_name)  # 测试集标签

    print(train_set.size())
    print(train_label.size())
    print(test_set.size())
    print(test_label.size())
    print(val_set.size())
    print(val_label.size())

    train_set = TensorDataset(train_set, train_label)  # 训练数据装成TensorDataset
    val_set = TensorDataset(val_set, val_label)  # 验证数据装成TensorDataset
    test_set = TensorDataset(test_set, test_label)  # 测试数据装成TensorDataset

    Train_DataLoader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=False)  # 使用DataLoader组织数据, 设置batch_size, 打乱数据
    Val_DataLoader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    Test_DataLoader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = TextCNN(vec_dim=embedding_dim, filter_num=filter_num, sentence_max_size=sentence_max_size,
                    label_size=label_size, kernel_list=kernel_list)  # text_cnn模型
    lamda = 0.7
    train_textcnn_model(model, Train_DataLoader, Val_DataLoader, num_epoch, lr, lamda, train_set, val_set)
    # torch.save(model.state_dict(), 'record/model.pth')
    est_textcnn_model(Test_DataLoader, lamda, test_set)
    compute_classification(Test_DataLoader, test_set)


if __name__ == '__main__':
    batch_size = 50
    embedding_dim = 300  # 词向量维度
    filter_num = 100  # 卷积器数量
    sentence_max_size = 40  # 句子最大长度
    label_size = 6  # 分类输出数量
    kernel_list = [3, 4, 5]  # 3种卷积器大小，每种100个
    num_epoch = 200  # 设置迭代次数
    lr = 0.02  # 学习率

    for i in range(10):
        # result_name = open("result.txt", "a")
        print("this is the " + str(i) + "th cross:")
        # print("this is the " + str(i) + "th cross:", file=result_name)
        # result_name.close()
        cross_model(i)

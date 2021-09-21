# -*- encoding: utf-8 -*-
import numpy as np
import gensim
import torch
import scipy.io as sio


def ReadData(path1, path2, path3, path4):   # 训练集文本、情绪分布, 测试集文本、情绪分布
    train_data = []
    train_sentence = np.load(path1)
    train_sentence = train_sentence.tolist()
    for line in train_sentence:
        matrix = np.zeros((40, 300), dtype=float)
        cnt = 0
        for word in line.split():
            try:
                embedding = np.array([model[word]])
            except Exception:  # 如果单词不在语料库中，在[-1,1]中随机定义
                embedding = np.random.uniform(0, 0, 300)
            matrix[cnt] = embedding
            cnt += 1
        train_data.append(matrix.reshape(1, 40, 300))

    train_data = np.array(train_data, dtype=float)
    data_size = train_data.shape[0]
    shuffled_index = np.random.permutation(data_size)
    train_data = train_data[shuffled_index]

    train_label_data = np.load(path2)
    train_label_data = train_label_data[shuffled_index]
    train_label_single_data = np.argmax(train_label_data, axis=1)

    td = train_label_single_data
    six_motion = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    for i in range(td.shape[0]):
        six_motion[td[i]].append(i)
    val_pos = []
    for i in range(6):
        ln = round(len(six_motion[i]) * 0.1)
        for j in six_motion[i][:ln]:
            val_pos.append(j)
        print(ln)
    train_pos = [w for w in range(data_size) if w not in val_pos]
    print(len(val_pos))
    val_pos = np.array(val_pos, dtype=int)
    train_pos = np.array(train_pos, dtype=int)


    val_set = train_data[val_pos]  # 验证集
    train_set = train_data[train_pos]  # 训练集

    val_label = train_label_data[val_pos]   # 验证集标签
    train_label = train_label_data[train_pos]  # 训练集标签

    val_label_single = train_label_single_data[val_pos]   # 验证集标签最大下标
    train_label_single = train_label_single_data[train_pos]  # 训练集标签最大下标

    val_set = torch.FloatTensor(val_set)   # 变为张量形式
    train_set = torch.FloatTensor(train_set)
    val_label = torch.FloatTensor(val_label)
    train_label = torch.FloatTensor(train_label)
    val_label_single = torch.LongTensor(val_label_single)
    train_label_single = torch.LongTensor(train_label_single)

    test_data = []
    test_sentence = np.load(path3)
    test_sentence = test_sentence.tolist()
    for line in test_sentence:
        matrix = np.zeros((40, 300), dtype=float)
        cnt = 0
        for word in line.split():
            try:
                embedding = np.array([model[word]])
            except Exception:  # 如果单词不在语料库中，在[-1,1]中随机定义
                embedding = np.random.uniform(0, 0, 300)
            matrix[cnt] = embedding
            cnt += 1
        test_data.append(matrix.reshape(1, 40, 300))

    test_set = np.array(test_data, dtype=float)
    test_label = np.load(path4)
    test_label_single = np.argmax(test_label, axis=1)

    test_set = torch.FloatTensor(test_set)
    test_label = torch.FloatTensor(test_label)
    test_label_single = torch.LongTensor(test_label_single)

    return train_set, train_label, train_label_single, val_set, val_label, val_label_single, test_set, test_label, test_label_single

def cross_save(k):
    file1name = "10-cross/cross" + str(k) + "/train_data.npy"  # 句子训练集
    file2name = "10-cross/cross" + str(k) + "/train_label.npy"  # 训练集情绪分布
    file3name = "10-cross/cross" + str(k) + "/test_data.npy"  # 句子测试集
    file4name = "10-cross/cross" + str(k) + "/test_label.npy"  # 测试集情绪分布
    print("Data is coming!")
    train_set, train_label, train_label_single, val_set, val_label, val_label_single, test_set, test_label, test_label_single = ReadData(file1name, file2name, file3name, file4name)
    '''
    train_set_name = "save4/train_set"+str(k)+".pth"
    train_label_name = "save4/train_label"+str(k)+".pth"
    torch.save(train_set, train_set_name)
    torch.save(train_label, train_label_name)

    val_set_name = "save4/val_set"+str(k)+".pth"
    val_label_name = "save4/val_label"+str(k)+".pth"
    torch.save(val_set, val_set_name)
    torch.save(val_label, val_label_name)

    test_set_name = "save4/test_set"+str(k)+".pth"
    test_label_name = "save4/test_label"+str(k)+".pth"
    torch.save(test_set, test_set_name)
    torch.save(test_label, test_label_name)
'''
    train_set = train_set.numpy()
    train_label = train_label.numpy()
    test_set = test_set.numpy()
    test_label = test_label.numpy()

    train_num = train_set.shape[0]
    test_num = test_set.shape[0]
    train_set = train_set.reshape(train_num, 12000)
    test_set = test_set.reshape(test_num, 12000)
    train_set = train_set[:, 0:8000]
    test_set = test_set[:, 0:8000]

    print(train_set.shape)
    print(test_set.shape)
    sio.savemat("mat/CBET" + str(k) +".mat", {'trainFeature': train_set, 'trainDistribution': train_label, 'trainNum': train_num,
                                'testFeature': test_set, 'testDistribution': test_label, 'testNum': test_num})


if __name__ == "__main__":
    print("ReadData...")
    model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)

    for i in range(10):
        print("this is the "+str(i)+"th cross:")
        cross_save(i)  # 将每一折W2V,并划分验证集
        print('\n')



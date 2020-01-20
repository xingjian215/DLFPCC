from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import keras

def load_data_at():
    resultfile = np.loadtxt('abnormal.txt', delimiter=',', dtype=np.str)
    # 主叫号码转独热，170位
    temp_ndarray1 = resultfile[:, 0:17]
    temp_ndarray1 = keras.utils.to_categorical(temp_ndarray1, 10)
    temp_ndarray1 = np.reshape(temp_ndarray1, (len(temp_ndarray1), 170))

    # 通话次数和被叫数量，2位
    temp_ndarray2 = resultfile[:, 17:18]
    temp_ndarray3 = resultfile[:, 18:19]
    temp_ndarray2 = temp_ndarray2.astype('float')
    temp_ndarray3 = temp_ndarray3.astype('float')
    temp_ndarray2 = (temp_ndarray2 - temp_ndarray2.min()) / (temp_ndarray2.max() - temp_ndarray2.min())
    temp_ndarray3 = (temp_ndarray3 - temp_ndarray3.min()) / (temp_ndarray3.max() - temp_ndarray3.min())
    temp_ndarray2 = np.reshape(temp_ndarray2, (len(temp_ndarray2), 1))
    temp_ndarray3 = np.reshape(temp_ndarray3, (len(temp_ndarray3), 1))

    # 被叫最大相似度和平均相似度，2位
    temp_ndarray4 = resultfile[:, 19:20]
    temp_ndarray5 = resultfile[:, 20:21]
    temp_ndarray4 = np.reshape(temp_ndarray4, (len(temp_ndarray4), 1))
    temp_ndarray5 = np.reshape(temp_ndarray5, (len(temp_ndarray5), 1))

    # 被叫平均时长和被叫地域数量，2位
    temp_ndarray6 = resultfile[:, 21:22]
    temp_ndarray7 = resultfile[:, 22:23]
    temp_ndarray6 = temp_ndarray6.astype('float')
    temp_ndarray7 = temp_ndarray7.astype('float')
    temp_ndarray6 = (temp_ndarray6 - temp_ndarray6.min()) / (temp_ndarray6.max() - temp_ndarray6.min())
    temp_ndarray7 = (temp_ndarray7 - temp_ndarray7.min()) / (temp_ndarray7.max() - temp_ndarray7.min())
    temp_ndarray6 = np.reshape(temp_ndarray6, (len(temp_ndarray6), 1))
    temp_ndarray7 = np.reshape(temp_ndarray7, (len(temp_ndarray7), 1))

    # 类型，1位
    temp_ndarray8 = resultfile[:, 23:24]
    temp_ndarray8 = keras.utils.to_categorical(temp_ndarray8, 2)

    # 组合temp_ndarray，最后一位为标签
    temp_ndarray = np.concatenate((temp_ndarray1, temp_ndarray2, temp_ndarray3, temp_ndarray4, temp_ndarray5, temp_ndarray6, temp_ndarray7, temp_ndarray8), axis=1)
    #temp_ndarray = np.concatenate((temp_ndarray1, temp_ndarray6, temp_ndarray7,temp_ndarray8), axis=1)
    # 随机temp_ndarray
    row_indices = np.random.permutation(temp_ndarray.shape[0])
    train_at = temp_ndarray[row_indices[0:6000], :]
    val_at = temp_ndarray[row_indices[6000:7000], :]
    test_at = temp_ndarray[row_indices[7000:8000], :]

    return train_at, val_at, test_at

def load_data_nt():
    resultfile = np.loadtxt('201812result_ok_no6_view(1601000).txt', delimiter=',', dtype=np.str)
    #resultfile = np.loadtxt('abnormal.txt', delimiter=',', dtype=np.str)
    # 主叫号码转独热，170位
    temp_ndarray1 = resultfile[:, 0:17]
    temp_ndarray1 = keras.utils.to_categorical(temp_ndarray1, 10)
    temp_ndarray1 = np.reshape(temp_ndarray1, (len(temp_ndarray1), 170))

    # 通话次数和被叫数量，2位
    temp_ndarray2 = resultfile[:, 17:18]
    temp_ndarray3 = resultfile[:, 18:19]
    temp_ndarray2 = temp_ndarray2.astype('float')
    temp_ndarray3 = temp_ndarray3.astype('float')
    temp_ndarray2 = (temp_ndarray2 - temp_ndarray2.min()) / (temp_ndarray2.max() - temp_ndarray2.min())
    temp_ndarray3 = (temp_ndarray3 - temp_ndarray3.min()) / (temp_ndarray3.max() - temp_ndarray3.min())
    temp_ndarray2 = np.reshape(temp_ndarray2, (len(temp_ndarray2), 1))
    temp_ndarray3 = np.reshape(temp_ndarray3, (len(temp_ndarray3), 1))

    # 被叫最大相似度和平均相似度，2位
    temp_ndarray4 = resultfile[:, 19:20]
    temp_ndarray5 = resultfile[:, 20:21]
    temp_ndarray4 = np.reshape(temp_ndarray4, (len(temp_ndarray4), 1))
    temp_ndarray5 = np.reshape(temp_ndarray5, (len(temp_ndarray5), 1))

    # 被叫平均时长和被叫地域数量，2位
    temp_ndarray6 = resultfile[:, 21:22]
    temp_ndarray7 = resultfile[:, 22:23]
    temp_ndarray6 = temp_ndarray6.astype('float')
    temp_ndarray7 = temp_ndarray7.astype('float')
    temp_ndarray6 = (temp_ndarray6 - temp_ndarray6.min()) / (temp_ndarray6.max() - temp_ndarray6.min())
    temp_ndarray7 = (temp_ndarray7 - temp_ndarray7.min()) / (temp_ndarray7.max() - temp_ndarray7.min())
    temp_ndarray6 = np.reshape(temp_ndarray6, (len(temp_ndarray6), 1))
    temp_ndarray7 = np.reshape(temp_ndarray7, (len(temp_ndarray7), 1))

    # 类型，1位
    temp_ndarray8 = resultfile[:, 23:24]
    temp_ndarray8 = keras.utils.to_categorical(temp_ndarray8, 2)

    # 组合temp_ndarray，最后一位为标签
    temp_ndarray = np.concatenate((temp_ndarray1, temp_ndarray2, temp_ndarray3, temp_ndarray4, temp_ndarray5, temp_ndarray6, temp_ndarray7, temp_ndarray8), axis=1)
    #temp_ndarray = np.concatenate((temp_ndarray1, temp_ndarray6, temp_ndarray7,temp_ndarray8), axis=1)

    # 随机temp_ndarray
    row_indices = np.random.permutation(temp_ndarray.shape[0])
    train_nt = temp_ndarray[row_indices[0:1200000], :]
    val_nt = temp_ndarray[row_indices[1200000:1201000], :]
    test_nt = temp_ndarray[row_indices[1201000:1202000], :]
    return train_nt, val_nt, test_nt

def load_data_predict(txtname):
    resultfile = np.loadtxt(txtname, delimiter=',', dtype=np.str)

    # 主叫号码转独热，170位
    temp_ndarray1 = resultfile[:, 0:17]
    temp_ndarray1 = keras.utils.to_categorical(temp_ndarray1, 10)
    temp_ndarray1 = np.reshape(temp_ndarray1, (len(temp_ndarray1), 170))

    # 通话次数和被叫数量，2位
    temp_ndarray2 = resultfile[:, 17:18]
    temp_ndarray3 = resultfile[:, 18:19]
    temp_ndarray2 = temp_ndarray2.astype('float')
    temp_ndarray3 = temp_ndarray3.astype('float')
    temp_ndarray2 = (temp_ndarray2 - temp_ndarray2.min()) / (temp_ndarray2.max() - temp_ndarray2.min())
    temp_ndarray3 = (temp_ndarray3 - temp_ndarray3.min()) / (temp_ndarray3.max() - temp_ndarray3.min())
    temp_ndarray2 = np.reshape(temp_ndarray2, (len(temp_ndarray2), 1))
    temp_ndarray3 = np.reshape(temp_ndarray3, (len(temp_ndarray3), 1))

    # 被叫最大相似度和平均相似度，2位
    temp_ndarray4 = resultfile[:, 19:20]
    temp_ndarray5 = resultfile[:, 20:21]
    temp_ndarray4 = np.reshape(temp_ndarray4, (len(temp_ndarray4), 1))
    temp_ndarray5 = np.reshape(temp_ndarray5, (len(temp_ndarray5), 1))

    # 被叫平均时长和被叫地域数量，2位
    temp_ndarray6 = resultfile[:, 21:22]
    temp_ndarray7 = resultfile[:, 22:23]
    temp_ndarray6 = temp_ndarray6.astype('float')
    temp_ndarray7 = temp_ndarray7.astype('float')
    temp_ndarray6 = (temp_ndarray6 - temp_ndarray6.min()) / (temp_ndarray6.max() - temp_ndarray6.min())
    temp_ndarray7 = (temp_ndarray7 - temp_ndarray7.min()) / (temp_ndarray7.max() - temp_ndarray7.min())
    temp_ndarray6 = np.reshape(temp_ndarray6, (len(temp_ndarray6), 1))
    temp_ndarray7 = np.reshape(temp_ndarray7, (len(temp_ndarray7), 1))

    # 类型，1位
    temp_ndarray8 = resultfile[:, 23:24]
    temp_ndarray8 = keras.utils.to_categorical(temp_ndarray8, 2)
    # 组合temp_ndarray
    temp_ndarray = np.concatenate((temp_ndarray1, temp_ndarray2, temp_ndarray3, temp_ndarray4, temp_ndarray5, temp_ndarray6, temp_ndarray7,), axis=1)


    return temp_ndarray,temp_ndarray8

def compose_data():
    #加载异常数据
    train_at, val_at, test_at=load_data_at()
    # 加载正常数据
    train_nt, val_nt, test_nt = load_data_nt()
    # 合并数据
    train_all = np.concatenate((train_at, train_nt), axis=0)
    val_all = np.concatenate((val_at, val_nt), axis=0)
    test_all = np.concatenate((test_at, test_nt), axis=0)
    # 拆分数据和标签
    data_first=0
    data_last = 50
    label_first=176
    label_last=178
    x_train = train_all[:, data_first:data_last]
    y_train = train_all[:, label_first:label_last]
    x_val = val_all[:, data_first:data_last]
    y_val = val_all[:, label_first:label_last]
    x_test = test_all[:, data_first:data_last]
    y_test = test_all[:, label_first:label_last]
    return (x_train, y_train), (x_val, y_val),(x_test, y_test)

def compose_predict_data(txtname):
    x_predict,y_predict=load_data_predict(txtname)
    return x_predict,y_predict
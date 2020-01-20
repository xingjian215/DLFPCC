import numpy as np
def predict(fit_model, x_test, y_test):
    # 对测试集进行预测
    predictions = fit_model.predict(x_test, batch_size=128, verbose=1)
    FP, FN, TP, TN = 0, 0, 0, 0
    result = []
    for i in predictions:
        if i[0] > i[1]:
            result.append(0)
        else:
            result.append(1)
    for i in range(len(result)):
        y_test_label = np.argmax(y_test[i])
        #y_test_label = y_test[i]
        if (result[i] == 1) and (result[i] == y_test_label):
            TP += 1
        elif (result[i] == 1) and (result[i] != y_test_label):
            FP += 1
        elif (result[i] == 0) and (result[i] == y_test_label):
            TN += 1
        else:
            FN += 1
    return TP, FP, TN, FN
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
import pandas as pd

#导入数据
def input_data_regisier(V_IMU,V_flag):
    df = pd.read_csv("D:\\GSTRIDE_database\\Database_register.csv",
                     delimiter=';', decimal=',', encoding='latin1')
    df = df.replace({',': '.'}, regex=True)
    rows, cols = df.shape
    print(f"行数: {rows}, 列数: {cols}")

    for i in range(163):
        V_IMU[i] = df.iloc[i + 2, 31:60]

    vflag = np.zeros((163), dtype=str)
    for i in range(163):
        vflag[i] = df.iloc[i + 2, 4]
        if (vflag[i] == 'N'):
            V_flag[i] = 0  # 非跌倒者标志位为0
        else:
            V_flag[i] = 1  # 跌倒者标志位为1

#评估数据
def assess(V_flag,V_flag_pred):
    TP = 0  # 准确地识别跌倒者
    TN = 0  # 准确地非跌倒者
    FP = 0  # 将非跌倒者错误地归类为跌倒者
    FN = 0  # 将跌倒者错误地归类为非跌倒者
    for i in range(len(V_flag)):
        if (V_flag[i] == V_flag_pred[i] and V_flag_pred[i] == 1):
            TP += 1
        if (V_flag[i] == V_flag_pred[i] and V_flag_pred[i] == 0):
            TN += 1
        if (V_flag[i] != V_flag_pred[i] and V_flag_pred[i] == 1):
            FP += 1
        if (V_flag[i] != V_flag_pred[i] and V_flag_pred[i] == 0):
            FN += 1
    acc = (TP + TN) / len(V_flag)
    sens= TP/(TP+FN)
    F1=2*TP/(2*TP+FP+FN)

    assess_value=[acc,sens,F1]
    return (assess_value)

#SVM训练
def SVM_Train(V_IMU,V_flag):
    svm_model = SVC(kernel='linear', C=1.0, gamma=2.5)
    acc_sum=0
    sens_sum = 0
    F1_sum = 0

    # V_train, V_test, V_flag_train, V_flag_test = train_test_split(V_IMU, V_flag,
    #                                                                 test_size=0.2,
    #                                                                 random_state=42)
    # svm_model = SVC(kernel='linear', C=1.0, gamma=2.5)
    # svm_model.fit(V_train, V_flag_train)
    # V_flag_test_pred = svm_model.predict(V_test)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(V_IMU):
        V_train, V_test = V_IMU[train_index], V_IMU[test_index]
        V_flag_train, V_flag_test = V_flag[train_index], V_flag[test_index]
        svm_model.fit(V_train, V_flag_train)

        V_flag_pred = svm_model.predict(V_test)
        assess_value=assess(V_flag_test, V_flag_pred)
        acc_sum=acc_sum+assess_value[1]
        sens_sum = acc_sum + assess_value[1]
        F1_sum = acc_sum + assess_value[1]

    acc_final=acc_sum/10
    sens_final = sens_sum / 10
    F1_final = F1_sum / 10
    print('acc=',acc_final,'sens=',sens_final,'F1=',F1_final)
    # return (svm_model)

#卡方检验
def chi2_caculate(V_IMU,V_flag):
    #计算最大最小值
    max = np.max(V_IMU)
    min = np.min(V_IMU)

    #根据最大最小值划分区间    num-1个区间，num个边界
    num = 4
    boundary = np.linspace(min, max, num)
    #根据边界值计算数据组别
    groups = np.digitize(V_IMU, boundary, right=True)

    #列联表        组别1         组别2         组别3         组别4         组别5
    #跌倒者       num_f[0]    num_f[1]      num_f[2]     num_f[3]    num_f[4]
    #非跌倒者     num_nf[0]   num_nf[1]     num_nf[2]    num_nf[3]   num_nf[4]
    num_f = np.zeros(num-1)
    num_nf = np.zeros(num-1)
    for i in range(163):
        if (V_flag[i] == 1):
            num_f[groups[i] - 1] += 1
        else:
            num_nf[groups[i] - 1] += 1

    data = np.array([[num_f],[num_nf]])
    chi2, p, dof, expected = chi2_contingency(data)
    return p
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, skew
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC, SVR
import pandas as pd


# 导入数据


def input_data_regisier(V_IMU, V_flag, V_TUG, V_GS, files_number_register):
    df = pd.read_csv("D:\\GSTRIDE_database\\Database_register.csv",
                     delimiter=';', decimal=',', encoding='latin1')
    df = df.replace({',': '.'}, regex=True)
    rows, cols = df.shape
    print(f"行数: {rows}, 列数: {cols}")

    for i in range(files_number_register):
        V_IMU[i] = df.iloc[i + 2, 31:60]
        try:
            V_TUG[i] = df.iloc[i + 2, 29]
        except ValueError:
            # 如果列中有无法转换为 float 的值，则将整列赋值为无穷大
            V_TUG[i] = np.inf

        V_GS[i] = df.iloc[i + 2, 17]

    vflag = np.zeros(files_number_register, dtype=str)
    for i in range(files_number_register):
        vflag[i] = df.iloc[i + 2, 4]
        if (vflag[i] == 'N'):
            V_flag[i] = 0  # 非跌倒者标志位为0
        else:
            V_flag[i] = 1  # 跌倒者标志位为1


# 评估数据
def assess(V_flag, V_flag_pred):
    TP = 0.0  # 准确地识别跌倒者
    TN = 0.0  # 准确地非跌倒者
    FP = 0.0  # 将非跌倒者错误地归类为跌倒者
    FN = 0.0  # 将跌倒者错误地归类为非跌倒者
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
    sens = TP / (TP + FN)
    F1 = 2 * TP / (2 * TP + FP + FN)

    assess_value = [acc, sens, F1]
    return (assess_value)

def SVM_Search(V_IMU, V_flag):
    pipeline = Pipeline([
        ('scaler', 'passthrough'),  # 占位符，稍后通过网格搜索替换为具体的归一化或标准化方法
        ('svm', SVC())  # 支持向量机
    ])

    # 定义参数网格

    #第一次返回C=0.1，G=0.01,模型得分0.71
    # param_grid = {
    #     'scaler': [StandardScaler(), MinMaxScaler()],
    #     'svm__kernel': ['linear', 'rbf'],  # 注意添加 'svm__' 前缀
    #     'svm__C': [0.1, 1, 10, 100],  # 正则化参数
    #     'svm__gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]  # 核函数系数
    # }

    # #第二次返回'svm__C': 0.2, 'svm__gamma': 0.05,模型得分0.7325
    # param_grid = {
    #     'scaler': [StandardScaler(), MinMaxScaler()],
    #     'svm__kernel': ['linear', 'rbf'],  # 注意添加 'svm__' 前缀
    #     'svm__C': [0.05,0.08,0.1,0.2,0.5],  # 正则化参数
    #     'svm__gamma': [0.005,0.008, 0.01,0.02,0.05]  # 核函数系数
    # }

    # 第三次返回'svm__C': 0.1, 'svm__gamma': 0.05,模型得分0.7325
    param_grid = {
        'scaler': [StandardScaler(), MinMaxScaler()],
        'svm__kernel': ['linear', 'rbf'],  # 注意添加 'svm__' 前缀
        'svm__C': [0.1,0.15, 0.2,0.25,0.3],  # 正则化参数
        'svm__gamma': [0.03,0.04,0.05,0.06,0.07]  # 核函数系数
    }

    # #Gait_Parameters_test
    # param_grid = {
    #     'scaler': [StandardScaler(), MinMaxScaler()],
    #     'svm__kernel': ['linear', 'rbf'],  # 注意添加 'svm__' 前缀
    #     'svm__C': [0.01,0.05,0.1,0.15,0.2],  # 正则化参数
    #     'svm__gamma': [0.05,0.06,0.07,0.08,0.09]  # 核函数系数
    # }

    # 使用GridSearchCV进行网格搜索-------指定的参数网格中搜索最佳超参数,使用 cv=10 进行 10 折交叉验证。
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=10, scoring='accuracy')
    # 进行网格搜索
    grid_search.fit(V_IMU, V_flag)
    # 获取最佳参数和最佳模型
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print("\n网格搜索结果")
    print("最佳参数:", best_params)
    print("最佳模型:", best_model)
    print("最佳模型得分:", grid_search.best_score_)
    # 返回最佳模型
    return best_model

def SVR_Search(V_IMU, V_flag):
    pipeline = Pipeline([
        ('scaler', 'passthrough'),  # 占位符，稍后通过网格搜索替换为具体的归一化或标准化方法
        ('svr', SVR())
            ])

    #
    param_grid = {
        'scaler': [StandardScaler(), MinMaxScaler()],
        'svr__kernel': ['linear', 'rbf'],
        'svr__C': [0.1, 1, 10, 100],  # 正则化参数
        'svr__gamma': [0.01,0.1, 1, 10, 100],  # 核函数系数
        'svr__epsilon': [0.01, 0.1, 0.5]  # ε-敏感损失函数
    }


    # 使用GridSearchCV进行网格搜索-------指定的参数网格中搜索最佳超参数,使用 cv=10 进行 10 折交叉验证。
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error')
    # 进行网格搜索
    grid_search.fit(V_IMU, V_flag)
    # 获取最佳参数和最佳模型
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print("\n网格搜索结果")
    print("最佳参数:", best_params)
    print("最佳模型:", best_model)
    print("最佳模型得分:", grid_search.best_score_)
    # 返回最佳模型
    return best_model

# 结果比较
def SVM_SVR_TUG_GS_Compare(svm_model,svr_model,V_IMU, V_flag,V_TUG,V_GS):
    # svm_model = SVC(kernel='rbf', C=0.1, gamma=0.01,random_state=42)
    # svr_model = SVR(kernel='rbf', C=0.1, gamma=0.04, epsilon=0.1)

    acc_SVM_sum = 0
    sens_SVM_sum = 0
    F1_SVM_sum = 0

    acc_SVR_sum = 0
    sens_SVR_sum = 0
    F1_SVR_sum = 0

    acc_TUG_sum = 0
    sens_TUG_sum = 0
    F1_TUG_sum = 0

    acc_GS_sum = 0
    sens_GS_sum = 0
    F1_GS_sum = 0

    # V_train, V_test, V_flag_train, V_flag_test = train_test_split(V_IMU, V_flag,
    #                                                                 test_size=0.2,
    #                                                                 random_state=42)
    # svm_model = SVC(kernel='linear', C=1.0, gamma=2.5)
    # svm_model.fit(V_train, V_flag_train)
    # V_flag_test_pred = svm_model.predict(V_test)

    kf = KFold(n_splits=10, shuffle=True)

    for train_index, test_index in kf.split(V_IMU):
        V_train, V_test = V_IMU[train_index], V_IMU[test_index]
        V_flag_train, V_flag_test = V_flag[train_index], V_flag[test_index]
        V_TUG_test=V_TUG[test_index]
        V_GS_test=V_GS[test_index]

        #归一化或标准化
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        V_train = scaler.fit_transform(V_train)
        V_test = scaler.transform(V_test)

        #SVM预测
        svm_model.fit(V_train, V_flag_train)
        V_flag_SVM_pred = svm_model.predict(V_test)

        assess_SVM_value = assess(V_flag_test, V_flag_SVM_pred)
        acc_SVM_sum = acc_SVM_sum + assess_SVM_value[0]
        sens_SVM_sum = sens_SVM_sum + assess_SVM_value[1]
        F1_SVM_sum = F1_SVM_sum + assess_SVM_value[2]

        # SVR预测
        svr_model.fit(V_train, V_flag_train)
        V_flag_SVR_pred = svr_model.predict(V_test)

        assess_SVR_value = assess(V_flag_test, V_flag_SVR_pred)
        acc_SVR_sum = acc_SVR_sum + assess_SVR_value[0]
        sens_SVR_sum = sens_SVR_sum + assess_SVR_value[1]
        F1_SVR_sum = F1_SVR_sum + assess_SVR_value[2]

        # TUG测试
        V_TUG_pred = np.zeros(len(V_flag_test))
        for i in range(len(V_flag_test)):
            if V_TUG_test[i] > 13.5:
                V_TUG_pred[i] = 1
        assess_TUG_value = assess(V_flag_test, V_TUG_pred)
        acc_TUG_sum = acc_TUG_sum + assess_TUG_value[0]
        sens_TUG_sum = sens_TUG_sum + assess_TUG_value[1]
        F1_TUG_sum = F1_TUG_sum + assess_TUG_value[2]

        # GS测试
        V_GS_pred = np.zeros(len(V_flag_test))
        for i in range(len(V_flag_test)):
            if V_GS_test[i] < 1:
                V_GS_pred[i] = 1
        assess_GS_value = assess(V_flag_test, V_GS_pred)
        acc_GS_sum = acc_GS_sum + assess_GS_value[0]
        sens_GS_sum = sens_GS_sum + assess_GS_value[1]
        F1_GS_sum = F1_GS_sum + assess_GS_value[2]

    print("\nSVM模型评估结果")
    meanacc_SVM = acc_SVM_sum / 10.0
    meansens_SVM = sens_SVM_sum / 10.0
    meanF1_SVM = F1_SVM_sum / 10.0
    print('meanacc_SVM=', meanacc_SVM)
    print('meansens_SVM', meansens_SVM)
    print('meanF1_SVM=', meanF1_SVM)

    print("\nSVR模型评估结果")
    meanacc_SVR = acc_SVR_sum / 10.0
    meansens_SVR = sens_SVR_sum / 10.0
    meanF1_SVR = F1_SVR_sum / 10.0
    print('meanacc_SVR=', meanacc_SVR)
    print('meansens_SVR', meansens_SVR)
    print('meanF1_SVR=', meanF1_SVR)

    print("\nTUG评估结果")
    meanacc_TUG = acc_TUG_sum / 10.0
    meansens_TUG = sens_TUG_sum / 10.0
    meanF1_TUG = F1_TUG_sum / 10.0
    print('meanacc_TUG=', meanacc_TUG)
    print('meansens_TUG', meansens_TUG)
    print('meanF1_TUG', meanF1_TUG)

    print("\nGS评估结果")
    meanacc_GS = acc_GS_sum / 10.0
    meansens_GS = sens_GS_sum / 10.0
    meanF1_GS = F1_GS_sum / 10.0
    print('meanacc_GS=', meanacc_GS)
    print('meansens_GS', meansens_GS)
    print('meanF1_GS=', meanF1_GS)
    # return (svm_model)


# 卡方检验
def chi2_calculate_Delete(V_IMU, V_flag, files_number_register):
    # 计算最大最小值
    max_val = np.max(V_IMU)
    min_val = np.min(V_IMU)
    num = 6
    # 根据最大最小值划分区间    num-1个区间，num个边界
    boundary = np.linspace(min_val, max_val, num)
    # 根据边界值计算数据组别-------groups 是一个和 V_IMU 长度相同的数组，其中每个元素是对应 V_IMU 数据点所属区间的编号
    groups = np.digitize(V_IMU, boundary, right=True)

    # 列联表        组别1         组别2         组别3         组别4         组别5
    # 跌倒者       num_f[0]    num_f[1]      num_f[2]     num_f[3]    num_f[4]
    # 非跌倒者     num_nf[0]   num_nf[1]     num_nf[2]    num_nf[3]   num_nf[4]
    num_f = np.zeros(num - 1)
    num_nf = np.zeros(num - 1)
    for i in range(files_number_register):
        if (V_flag[i] == 1):
            num_f[groups[i] - 1] += 1
        else:
            num_nf[groups[i] - 1] += 1
    data = np.array([num_f, num_nf])

    valid_columns = np.any(data > 0, axis=0)  # 检测非全零列
    data = data[:, valid_columns]  # 保留非全零列
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("列联表中没有有效数据，无法进行卡方检验。")
    chi2, p, dof, expected = chi2_contingency(data)

    return p, num


def chi2_calculate_Search(V_IMU, V_flag, files_number_register):
    # 计算最大最小值
    max_val = np.max(V_IMU)
    min_val = np.min(V_IMU)
    best_chi2 =0
    best_p = 1
    best_num = None

    for num in range(3, 21):
        boundary = np.linspace(min_val, max_val, num)
        groups = np.digitize(V_IMU, boundary, right=True)

        num_f = np.zeros(num - 1)
        num_nf = np.zeros(num - 1)
        for i in range(files_number_register):
            if (V_flag[i] == 1):
                num_f[groups[i] - 1] += 1
            else:
                num_nf[groups[i] - 1] += 1
        data = np.array([num_f, num_nf])

        if np.any(num_f + num_nf < 5):
            continue

        chi2, p, dof, expected = chi2_contingency(data)
        if chi2 > best_chi2:
            best_p = p
            best_chi2 = chi2
            best_num = num

    if best_num is None:
        best_p = 1
    return best_p, best_num


# 存储所有文件的矩阵
def input_data_AccGyr():
    AccGyr = {}  # 创建一个空字典
    folder_path = "D:\\GSTRIDE_database\\Test_recordings_calibrated"
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        name_without_ext = os.path.splitext(file_name)[0]  # 去掉 .txt
        file_path = os.path.join(folder_path, file_name)
        # 读取文件并将其转换为矩阵
        matrix = np.loadtxt(file_path, skiprows=1)  # 假设数据以空格分隔
        AccGyr[name_without_ext] = matrix
    return AccGyr

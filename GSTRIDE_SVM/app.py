import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, skew
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
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


# 导入加速度,角速度
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


def ROC(y_true, y_pred):

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    gmeans = np.sqrt(tpr * (1 - fpr))
    best_idx = np.argmax(gmeans)
    best_threshold = thresholds[best_idx]
    # print(f"最佳阈值: {best_threshold}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random guess')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)

    return best_threshold


#SVM网格搜索
def SVM_Search(V_IMU, V_flag):
    pipeline = Pipeline([
        ('scaler', 'passthrough'),  # 占位符，稍后通过网格搜索替换为具体的归一化或标准化方法
        # ('dim_reduce', PCA()),  # 注意这里叫 dim_reduce
        ('svm', SVC())  # 支持向量机
    ])

    # 第三次返回'svm__C': 0.1, 'svm__gamma': 0.05,模型得分0.7325
    param_grid = {
        'scaler': [StandardScaler(), MinMaxScaler()],
        # 'dim_reduce': [PCA()],
        # 'dim_reduce__n_components': [2, 3, 4],
        'svm__kernel': ['linear', 'rbf'],
        'svm__C': [0.05,0.1,0.15,0.2,0.25],
        'svm__gamma': [0.01,0.05,0.1,0.5,1]
    }

    # 使用GridSearchCV进行网格搜索-------指定的参数网格中搜索最佳超参数,使用 cv=10 进行 10 折交叉验证。
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=10, scoring='accuracy')
    # 进行网格搜索
    grid_search.fit(V_IMU, V_flag)
    # 获取最佳参数和最佳模型
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print("\nSVM网格搜索结果")
    print("最佳参数:", best_params)
    print("最佳模型得分:", grid_search.best_score_)

    # print("\n所有参数组合得分如下：")
    # results = grid_search.cv_results_
    # for mean_score, std_score, params in zip(results['mean_test_score'], results['std_test_score'], results['params']):
    #     print(f"参数: {params} | 平均准确率: {mean_score:.4f} (+/- {std_score:.4f})")

    # 返回最佳模型
    return best_model


#SVR网格搜索
def SVR_Search(V_IMU, V_flag):
    pipeline = Pipeline([
        ('scaler', 'passthrough'),  # 占位符，稍后通过网格搜索替换为具体的归一化或标准化方法
        # ('dim_reduce', PCA()),  # 注意这里叫 dim_reduce
        ('svr', SVR())
            ])

    param_grid = {
        'scaler': [StandardScaler(), MinMaxScaler()],
        # 'dim_reduce': [PCA()],
        # 'dim_reduce__n_components': [2, 3, 4],
        'svr__kernel': ['linear', 'rbf'],
        'svr__C': [0.1,0.15,0.2,0.25,0.3,0.35],
        'svr__gamma': [0.01,0.05,0.1,0.5,1],
        'svr__epsilon': [0.2,0.25,0.3,0.35]  # ε-敏感损失函数
    }

    # 使用GridSearchCV进行网格搜索-------指定的参数网格中搜索最佳超参数,使用 cv=10 进行 10 折交叉验证。
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error')
    # 进行网格搜索
    grid_search.fit(V_IMU, V_flag)
    # 获取最佳参数和最佳模型
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print("\nSVR网格搜索结果")
    print("最佳参数:", best_params)
    print("最佳模型得分:", grid_search.best_score_)

    # print("\n所有参数组合得分如下：")
    # results = grid_search.cv_results_
    # for mean_score, std_score, params in zip(results['mean_test_score'], results['std_test_score'], results['params']):
    #     print(f"参数: {params} | 平均准确率: {mean_score:.4f} (+/- {std_score:.4f})")

    # 返回最佳模型
    return best_model


# TUG测试
def TUG_test(V_flag,V_TUG):
    TUG_shold=13
    while TUG_shold<=15:
        V_TUG_pred = np.zeros(len(V_flag))
        for i in range(len(V_flag)):
            if V_TUG[i] > TUG_shold:
                V_TUG_pred[i] = 1
        assess_TUG_value = assess(V_flag, V_TUG_pred)
        print("\nTUG评估结果",TUG_shold)
        print('meanacc_TUG=',assess_TUG_value[0])
        print('meansens_TUG',assess_TUG_value[1])
        print('meanF1_TUG',assess_TUG_value[2])
        TUG_shold = TUG_shold + 0.2

# GS测试
def GS_test(V_flag,V_GS):
    GS_shold=0.5
    while GS_shold <= 1.05:
        V_GS_pred = np.zeros(len(V_flag))
        for i in range(len(V_flag)):
            if V_GS[i] < GS_shold:
                V_GS_pred[i] = 1
        assess_GS_value = assess(V_flag, V_GS_pred)
        print("\nGS评估结果",GS_shold)
        print('meanacc_GS=', assess_GS_value[0])
        print('meansens_GS', assess_GS_value[1])
        print('meanF1_GS=', assess_GS_value[2])
        GS_shold = GS_shold + 0.05


# 结果比较
def SVM_Train(svm_model,V_IMU, V_flag):
    # svm_model = SVC(kernel='rbf', C=0.1, gamma=0.01,random_state=42)
    # svr_model = SVR(kernel='rbf', C=0.1, gamma=0.04, epsilon=0.1)

    acc_SVM_sum = 0
    sens_SVM_sum = 0
    F1_SVM_sum = 0

    # V_train, V_test, V_flag_train, V_flag_test = train_test_split(V_IMU, V_flag,
    #                                                                 test_size=0.2,
    #                                                                 random_state=42)
    # svm_model = SVC(kernel='linear', C=1.0, gamma=2.5)
    # svm_model.fit(V_train, V_flag_train)
    # V_flag_test_pred = svm_model.predict(V_test)

    kf = StratifiedKFold(n_splits=10, shuffle=True,random_state=42)

    for train_index, test_index in kf.split(V_IMU,V_flag):
        V_SVM_train, V_SVM_test = V_IMU[train_index], V_IMU[test_index]
        V_flag_train, V_flag_test = V_flag[train_index], V_flag[test_index]

        # #SVM预测
        # #归一化或标准化
        # scaler = StandardScaler()
        # V_SVM_train = scaler.fit_transform(V_SVM_train)
        # V_SVM_test = scaler.transform(V_SVM_test)

        # # LDA降维
        # lda = LinearDiscriminantAnalysis(n_components=1)
        # V_SVM_train = lda.fit_transform(V_SVM_train, V_flag_train)
        # V_SVM_test = lda.transform(V_SVM_test)

        # # PCA降维
        # pca = PCA(n_components=5)
        # V_SVM_train = pca.fit_transform(V_SVM_train)
        # V_SVM_test = pca.transform(V_SVM_test)

        svm_model.fit(V_SVM_train, V_flag_train)
        V_flag_SVM_pred = svm_model.predict(V_SVM_test)

        assess_SVM_value = assess(V_flag_test, V_flag_SVM_pred)
        acc_SVM_sum = acc_SVM_sum + assess_SVM_value[0]
        sens_SVM_sum = sens_SVM_sum + assess_SVM_value[1]
        F1_SVM_sum = F1_SVM_sum + assess_SVM_value[2]


    print("\nSVM模型评估结果")
    meanacc_SVM = acc_SVM_sum / 10.0
    meansens_SVM = sens_SVM_sum / 10.0
    meanF1_SVM = F1_SVM_sum / 10.0
    print('meanacc_SVM=', meanacc_SVM)
    print('meansens_SVM', meansens_SVM)
    print('meanF1_SVM=', meanF1_SVM)

    # return (svm_model)


def SVR_Train(svr_model,V_IMU, V_flag):
    # 归一化或标准化
    acc_SVR_sum = 0
    sens_SVR_sum = 0
    F1_SVR_sum = 0

    all_test_preds = []
    all_test_flags = []

    kf = StratifiedKFold(n_splits=10, shuffle=True,random_state=42)

    for train_index, test_index in kf.split(V_IMU, V_flag):
        V_SVR_train, V_SVR_test = V_IMU[train_index], V_IMU[test_index]
        V_flag_train, V_flag_test = V_flag[train_index], V_flag[test_index]

        # scaler = StandardScaler()
        # V_SVR_train = scaler.fit_transform(V_SVR_train)
        # V_SVR_test = scaler.transform(V_SVR_test)

        # # LDA降维
        # lda = LinearDiscriminantAnalysis(n_components=1)
        # V_SVR_train = lda.fit_transform(V_SVR_train, V_flag_train)
        # V_SVR_test = lda.transform(V_SVR_test)

        # # PCA降维
        # pca = PCA(n_components=3)
        # V_SVR_train = pca.fit_transform(V_SVR_train)
        # V_SVR_test = pca.transform(V_SVR_test)

        svr_model.fit(V_SVR_train, V_flag_train)
        V_SVR_pred = svr_model.predict(V_SVR_test)

        V_SVR_pred = np.array(V_SVR_pred)
        th_value = ROC(V_flag_test, V_SVR_pred)
        V_flag_SVR_pred = (V_SVR_pred >= th_value).astype(int)

        all_test_preds.extend(V_SVR_pred)
        all_test_flags.extend(V_flag_test)

        assess_SVR_value = assess(V_flag_test, V_flag_SVR_pred)
        acc_SVR_sum = acc_SVR_sum + assess_SVR_value[0]
        sens_SVR_sum = sens_SVR_sum + assess_SVR_value[1]
        F1_SVR_sum = F1_SVR_sum + assess_SVR_value[2]

    all_test_preds = np.array(all_test_preds)
    all_test_flags = np.array(all_test_flags)
    all_threshold = ROC(all_test_flags, all_test_preds)

    print("\nSVR模型评估结果")
    meanacc_SVR = acc_SVR_sum / 10.0
    meansens_SVR = sens_SVR_sum / 10.0
    meanF1_SVR = F1_SVR_sum / 10.0
    print('meanacc_SVR=', meanacc_SVR)
    print('meansens_SVR', meansens_SVR)
    print('meanF1_SVR=', meanF1_SVR)

    return all_threshold

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

#卡方检验寻找最优分组
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




from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC, SVR
from GSTRIDE_SVM.app import chi2_calculate_Delete, chi2_calculate_Search, assess,  SVM_Search, \
    SVR_Search, SVR_Train, TUG_test, SVM_Train, GS_test
from data import *


# 卡方检验-----------将p小于0.05的数据重组为一个矩阵
Gait_Parameters_x2 = np.empty((Gait_Parameters.shape[0], 0))
Gait_Parameters_name_x2 = []
p_x2=[]
row1, column1 = Gait_Parameters.shape
print("卡方检验结果")
for i in range(column1):
    # p,num=chi2_calculate_Delete(Gait_Parameters[:, i],V_flag,files_number_register)
    p, num = chi2_calculate_Search(Gait_Parameters[:, i], V_flag, files_number_register)
    if p< 0.05:
        Gait_Parameters_x2 = np.column_stack((Gait_Parameters_x2, Gait_Parameters[:, i]))
        Gait_Parameters_name_x2.append(Gait_Parameters_name[i])
        p_x2.append(p)
    else:
        print("分组数为", num, "p=", p, "步态参数", Gait_Parameters_name[i])


# 皮尔逊相关系数-----------将c大于0.9的数据删去
Gait_Parameters_final = Gait_Parameters_x2
Gait_Parameters_name_final = Gait_Parameters_name_x2
row2, column2 = Gait_Parameters_x2.shape
print("\n皮尔逊相关系数计算结果")
for i in range(column2 - 1):
    selected_prs = []  # 存储待删除数据
    for j in range(i + 1, column2):
        c,prs= pearsonr(Gait_Parameters_final[:, i], Gait_Parameters_final[:, j])
        if abs(c) > 0.9:
            print('c=', c, '步态参数', Gait_Parameters_name_final[i], '&', Gait_Parameters_name_final[j])
            #删除更不相关数据
            if p_x2[i] < p_x2[j]:
                selected_prs.append(j)  # 删除特征 j
                print("删除数据",Gait_Parameters_name_final[j])
            else:
                selected_prs.append(i)  # 删除特征 i
                print("删除数据", Gait_Parameters_name_final[i])
                break
            # selected_prs.append(j)
            row2, column2 = Gait_Parameters_final.shape

    Gait_Parameters_final = np.delete(Gait_Parameters_final, selected_prs, axis=1)
    Gait_Parameters_name_final = np.delete(Gait_Parameters_name_final, selected_prs)
    row2, column2 = Gait_Parameters_final.shape
    # Gait_Parameters_name_final = [name for idx, name in enumerate(Gait_Parameters_name_x2) if idx not in selected_prs]

# column4=Gait_Parameters_name_final.shape
# print(column3)
# print(column4)


#只将15s加速度、角速度用于计算
row3,column3=Gait_Parameters_final.shape
Gait_Parameters_final[:,column3-5]=meanAccX_Part.ravel()
Gait_Parameters_final[:,column3-4]=meanAccY_Part.ravel()
Gait_Parameters_final[:,column3-3]=meanGyrX_Part.ravel()
Gait_Parameters_final[:,column3-2]=meanGyrY_Part.ravel()
Gait_Parameters_final[:,column3-1]=meanGyrZ_Part.ravel()


svm_model=SVM_Search(Gait_Parameters_final, V_flag)
svr_model=SVR_Search(Gait_Parameters_final, V_flag)


#Permutation Importance-------------随机打乱验证数据某一列的值，保持目标列以及其它列的数据不变，观察对预测准确率产生的影响
# n_repeats=30：每个特征打乱的重复次数，越高结果越稳定但计算时间越长
PI_SVM_result = permutation_importance(svm_model, Gait_Parameters_final, V_flag, n_repeats=30, random_state=42, scoring='accuracy')
PI_SVR_result = permutation_importance(svr_model, Gait_Parameters_final, V_flag, n_repeats=30, random_state=42, scoring='neg_mean_squared_error')
print("\nSVM删除PI")
for i in range(column3):
    if PI_SVM_result.importances_mean[i]<-0.008:
        print(Gait_Parameters_name_final[i],PI_SVM_result.importances_mean[i])

SVM_Importances = PI_SVM_result.importances_mean
SVM_Delete = np.where(SVM_Importances < -0.008)
Gait_Parameters_SVM = np.delete(Gait_Parameters_final, SVM_Delete, axis=1)


print("SVR删除PI")
for i in range(column3):
    if PI_SVR_result.importances_mean[i]<0.001:
        print(Gait_Parameters_name_final[i],PI_SVR_result.importances_mean[i])

SVR_Importances = PI_SVR_result.importances_mean
SVR_Delete = np.where(SVR_Importances < 0.001)
Gait_Parameters_SVR = np.delete(Gait_Parameters_final, SVR_Delete, axis=1)


#模型训练
svm_model=SVM_Search(Gait_Parameters_SVM, V_flag)
# print(Gait_Parameters_SVM.shape)
svr_model=SVR_Search(Gait_Parameters_SVR, V_flag)

SVM_Train(svm_model,Gait_Parameters_SVM, V_flag)
# TUG_test(V_flag,V_TUG)
# GS_test(V_flag,V_GS)
all_threshold=SVR_Train(svr_model,Gait_Parameters_SVR, V_flag)
print("SVR全局阈值=",all_threshold)
plt.show()




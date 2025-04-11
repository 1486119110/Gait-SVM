import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from app import SVM_Train, assess, chi2_caculate
from data import *

#卡方检验-----------将p小于0.05的数据重组为一个矩阵
Gait_Parameters_x2=Gait_Parameters[:, 0]
Gait_Parameters_name_x2=[Gait_Parameters_name[0]]
row1,column1=Gait_Parameters.shape
print("卡方检验结果")
for i in range(column1-1):
    p=chi2_caculate(Gait_Parameters[:, i],V_flag)
    if p < 0.05:
        Gait_Parameters_x2=np.column_stack((Gait_Parameters_x2,Gait_Parameters[:, i]))
        Gait_Parameters_name_x2.append(Gait_Parameters_name[i])
    else:
        print('p=',p,'步态参数',Gait_Parameters_name[i])


#皮尔逊相关系数-----------将c大于0.9的数据删去
Gait_Parameters_final=Gait_Parameters_x2
Gait_Parameters_name_final=Gait_Parameters_name_x2
row2,column2=Gait_Parameters_x2.shape
print("皮尔逊相关系数计算结果")
for i in range(column2-2):
    selected_prs = []  # 存储待删除数据
    for j in range(i+1,column2-1):
        c,prs = pearsonr(Gait_Parameters_final[:, i],Gait_Parameters_final[:, j])
        if abs(c) > 0.9:
            selected_prs.append(j)
            print('c=', c, '步态参数',Gait_Parameters_name_final[i],'&',Gait_Parameters_name_final[j])
    Gait_Parameters_final = np.delete(Gait_Parameters_final, selected_prs, axis=1)
    Gait_Parameters_name_final = np.delete(Gait_Parameters_name_final, selected_prs)
    row2, column2 = Gait_Parameters_final.shape
    # Gait_Parameters_name_final = [name for idx, name in enumerate(Gait_Parameters_name_x2) if idx not in selected_prs]

# #归一化
scaler = StandardScaler()
Gait_train_scaler = scaler.fit_transform(Gait_Parameters_final)

#svm训练
print("模型评估结果")
svm_model=SVM_Train(Gait_train_scaler,V_flag)
# V_flag_pred = svm_model.predict(Gait_train_scaler)
# print("模型评估结果")
# assess(V_flag,V_flag_pred)










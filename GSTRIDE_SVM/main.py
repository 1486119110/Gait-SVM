from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression

from GSTRIDE_SVM.app import chi2_calculate_Delete, chi2_calculate_Search, assess, SVM_SVR_TUG_GS_Compare, SVM_Search, \
    SVR_Search
from data import *

# 卡方检验-----------将p小于0.05的数据重组为一个矩阵
Gait_Parameters_x2 = np.empty((Gait_Parameters.shape[0], 0))
Gait_Parameters_name_x2 = []
p_x2=[]
row1, column1 = Gait_Parameters.shape
print("卡方检验结果")
for i in range(column1):
    #p,num=chi2_calculate_Delete(Gait_Parameters[:, i],V_flag,files_number_register)
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

# row3,column3=Gait_Parameters_final.shape
# column4=Gait_Parameters_name_final.shape
# print(column3)
# print(column4)


svm_model=SVM_Search(Gait_Parameters_final, V_flag)
svr_model=SVR_Search(Gait_Parameters_final, V_flag)
SVM_SVR_TUG_GS_Compare(svm_model,svr_model,Gait_Parameters_final, V_flag,V_TUG,V_GS)

# V_flag_pred = svm_model.predict(Gait_train_scaler)
# print("模型评估结果")
# assess(V_flag,V_flag_pred)


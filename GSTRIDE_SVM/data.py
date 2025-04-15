import math
import numpy as np
from GSTRIDE_SVM.app import input_data_regisier, input_data_AccGyr

# 导入三轴加速度，三轴角速度
AccGyr = input_data_AccGyr()
files = list(AccGyr.keys())  # 获取所有文件名
files_number = len(files)
meanAccX = np.zeros((files_number, 1))
meanAccY = np.zeros((files_number, 1))
meanAccZ = np.zeros((files_number, 1))
meanAcc = np.zeros((files_number, 1))

meanGyrX = np.zeros((files_number, 1))
meanGyrY = np.zeros((files_number, 1))
meanGyrZ = np.zeros((files_number, 1))
meanGyr = np.zeros((files_number, 1))

# print(AccGyr["V001"][0][0])

for i, file_name in enumerate(files):
    data_AccGyr = AccGyr[file_name]  # 当前文件的矩阵数据
    row_AccGyr, column_AccGyr = data_AccGyr .shape

    # 计算加速度和陀螺仪的均值（按列操作）
    meanAccX[i] = np.mean(data_AccGyr[:, 0])  # 第一列为 AccX
    meanAccY[i] = np.mean(data_AccGyr[:, 1])  # 第二列为 AccY
    meanAccZ[i] = np.mean(data_AccGyr[:, 2])  # 第三列为 AccZ
    # meanAcc[i] = np.sqrt(meanAccX[i]**2 + meanAccY[i]**2 + meanAccZ[i]**2)  # 总加速度均值

    meanGyrX[i] = np.mean(data_AccGyr[:, 3])  # 第四列为 GyrX
    meanGyrY[i] = np.mean(data_AccGyr[:, 4])  # 第五列为 GyrY
    meanGyrZ[i] = np.mean(data_AccGyr[:, 5])  # 第六列为 GyrZ
    # meanGyr[i] = np.sqrt(meanGyrX[i]**2 + meanGyrY[i]**2 + meanGyrZ[i]**2)  # 总陀螺仪均值

files_number_register = 158

# 导入数据，数据存储在给定矩阵中   跌倒者标志位为0    跌倒者标志位为1
V_IMU = np.zeros((files_number_register, 29))
V_flag = np.zeros(files_number_register)
V_TUG = np.zeros((files_number_register, 1))
V_GS = np.zeros((files_number_register, 1))
input_data_regisier(V_IMU, V_flag, V_TUG, V_GS, files_number_register)

# 步行距离--------------数据变差
Distance = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    Distance[i][0] = V_IMU[i][0]

# Strides--------完整的步态周期数----------数据变差
Strides = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    Strides[i][0] = V_IMU[i][2]

# meanStride_Velocity-------同一只脚连续两次接触地面之间的动作
meanStride_Velocity = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    meanStride_Velocity[i][0] = V_IMU[i][21] / V_IMU[i][3]

# meanClearance
meanClearance = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    meanClearance[i][0] = V_IMU[i][27]

# mean%Loading----------落地后，把身体重量慢慢压到脚上
meanLoading = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    meanLoading[i][0] = V_IMU[i][5]

# mean%Pushing-----------脚离地了，摆到前面准备下一步
meanPushing = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    meanPushing[i][0] = V_IMU[i][9]

# mean%Swing------------脚掌发力推地，把身体往前送
meanSwing = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    meanSwing[i][0] = V_IMU[i][11]

# mean%FootFlat----------脚掌完全接触地面的那段时间----------数据变差
meanFootFlat = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    meanFootFlat[i][0] = V_IMU[i][7]

# meanPitchHS----------Heel-Strike俯仰角的均值
meanPitchHS = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    meanPitchHS[i][0] = V_IMU[i][15]

# meanStride_Length-------变异系数CV=标准差与均值的比值
meanStride_Length = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    meanStride_Length[i][0] = V_IMU[i][21]

# meanStep_Speed-------单只脚从一个落地到另一只脚落地之间
meanStep_Speed = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    meanStep_Speed[i][0] = V_IMU[i][19]

# meanCadence
meanCadence = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    meanCadence[i][0] = V_IMU[i][17]

# meanPitchToeOff----------脚尖离地俯仰角的均值
meanPitchToeOff = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    meanPitchToeOff[i][0] = V_IMU[i][13]

# meanPathLength3D
meanPathLength3D = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    meanPathLength3D[i][0] = V_IMU[i][23]

# meanPathLength2D
meanPathLength2D = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    meanPathLength2D[i][0] = V_IMU[i][25]

# CVStride_Velocity
CVStride_Velocity = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    CVStride_Velocity[i][0] = (V_IMU[i][22] / V_IMU[i][4]) / (V_IMU[i][21] / V_IMU[i][3]) * 100

# CVClearance
CVClearance = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    CVClearance[i][0] = V_IMU[i][28] / V_IMU[i][27] * 100

# CV%Loading
CVLoading = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    CVLoading[i][0] = V_IMU[i][6] / V_IMU[i][5] * 100

# CV%Pushing
CVPushing = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    CVPushing[i][0] = V_IMU[i][10] / V_IMU[i][9] * 100

# CV%Swing
CVSwing = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    CVSwing[i][0] = V_IMU[i][12] / V_IMU[i][11] * 100

# CV%FootFlat----------落地后，把身体重量慢慢压到脚上
CVFootFlat = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    CVFootFlat[i][0] = V_IMU[i][8] / V_IMU[i][7] * 100

# CVPitchHS----------Heel-Strike脚后跟着地俯仰角的变异系数
CVPitchHS = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    CVPitchHS[i][0] = V_IMU[i][16] / V_IMU[i][15] * 100

# CVStride_Length-------变异系数CV=标准差与均值的比值
CVStride_Length = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    CVStride_Length[i][0] = V_IMU[i][22] / V_IMU[i][21] * 100

# CVStep_Speed
CVStep_Speed = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    CVStep_Speed[i][0] = V_IMU[i][20] / V_IMU[i][19] * 100

# CVCadence
CVCadence = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    CVCadence[i][0] = V_IMU[i][18] / V_IMU[i][17] * 100

# CVPitchToeOff----------脚尖离地俯仰角的变异系数------------数据变差
CVPitchToeOff = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    CVPitchToeOff[i][0] = V_IMU[i][14] / V_IMU[i][13] * 100

# CVPathLength3D
CVPathLength3D = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    CVPathLength3D[i][0] = V_IMU[i][24] / V_IMU[i][23] * 100

# CVPathLength2D
CVPathLength2D = np.zeros((files_number_register, 1))
for i in range(files_number_register):
    CVPathLength2D[i][0] = V_IMU[i][26] / V_IMU[i][25] * 100

Gait_Parameters = np.column_stack((Distance,                Strides,                meanStride_Velocity,
                                   meanClearance,           meanLoading,            meanPushing,
                                   meanSwing,               meanFootFlat,           meanPitchHS,
                                   meanStride_Length,       meanStep_Speed,         meanCadence,
                                   meanPitchToeOff,         meanPathLength3D,       meanPathLength2D,
                                   CVStride_Velocity,       CVClearance,            CVLoading,
                                   CVPushing,               CVSwing,                CVFootFlat,
                                   CVPitchHS,               CVStride_Length,        CVStep_Speed,
                                   CVCadence,               CVPitchToeOff,          CVPathLength3D,
                                   CVPathLength2D,
                                   meanAccX,meanAccY,meanAccZ,meanGyrX,meanGyrY,meanGyrZ
                                   ))

Gait_Parameters_name = [            "Distance",             "Strides",              "meanStride_Velocity",
                                    "meanClearance",        "meanLoading",          "meanPushing",
                                    "meanSwing",            "meanFootFlat",         "meanPitchHS",
                                    "meanStride_Length",    "meanStep_Speed",       "meanCadence",
                                    "meanPitchToeOff",      "meanPathLength3D",     "meanPathLength2D",
                                    "CVStride_Velocity",    "CVClearance",          "CVLoading",
                                    "CVPushing",            "CVSwing",              "CVFootFlat",
                                    "CVPitchHS",            "CVStride_Length",      "CVStep_Speed",
                                    "CVCadence",            "CVPitchToeOff",        "CVPathLength3D",
                                    "CVPathLength2D",
                                    "meanAccX","meanAccY","meanAccZ","meanGyrX","meanGyrY","meanGyrZ"
                        ]

Gait_Parameters_test= np.column_stack(( meanStride_Velocity,     meanClearance,           CVStride_Length,
                                        meanLoading,             CVPitchHS,               CVSwing,
                                        CVStride_Velocity,       meanPushing,             CVPathLength3D,
                                        meanAccX,meanAccY,meanAccZ,meanGyrX,meanGyrY,meanGyrZ))

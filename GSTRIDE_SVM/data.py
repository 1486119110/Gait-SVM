import numpy as np
from GSTRIDE_SVM.app import input_data_regisier

#导入数据，数据存储在给定矩阵中   跌倒者标志位为0    跌倒者标志位为1
V_IMU = np.zeros((163, 29))
V_flag = np.zeros((163))
input_data_regisier(V_IMU,V_flag)


#步行距离--------------数据变差
Distance=np.zeros((163,1))
for i in range(163):
    Distance[i][0]=V_IMU[i][0]

#Strides--------完整的步态周期数----------数据变差
Strides=np.zeros((163,1))
for i in range(163):
    Strides[i][0]=V_IMU[i][2]

#meanStride_Velocity-------同一只脚连续两次接触地面之间的动作
meanStride_Velocity=np.zeros((163,1))
for i in range(163):
    meanStride_Velocity[i][0]=V_IMU[i][21]/V_IMU[i][3]

#meanClearance
meanClearance=np.zeros((163,1))
for i in range(163):
    meanClearance[i][0] = V_IMU[i][27]

#mean%Loading----------落地后，把身体重量慢慢压到脚上
meanLoading=np.zeros((163,1))
for i in range(163):
    meanLoading[i][0] = V_IMU[i][5]

#mean%Pushing-----------脚离地了，摆到前面准备下一步
meanPushing=np.zeros((163,1))
for i in range(163):
    meanPushing[i][0]=V_IMU[i][9]

#mean%Swing------------脚掌发力推地，把身体往前送
meanSwing=np.zeros((163,1))
for i in range(163):
    meanSwing[i][0] = V_IMU[i][11]

#mean%FootFlat----------脚掌完全接触地面的那段时间----------数据变差
meanFootFlat=np.zeros((163,1))
for i in range(163):
    meanFootFlat[i][0] = V_IMU[i][7]

#meanPitchHS----------Heel-Strike俯仰角的均值
meanPitchHS=np.zeros((163,1))
for i in range(163):
    meanPitchHS[i][0]=V_IMU[i][15]

#meanStride_Length-------变异系数CV=标准差与均值的比值
meanStride_Length=np.zeros((163,1))
for i in range(163):
    meanStride_Length[i][0]=V_IMU[i][21]

#meanStep_Speed-------单只脚从一个落地到另一只脚落地之间
meanStep_Speed=np.zeros((163,1))
for i in range(163):
    meanStep_Speed[i][0]=V_IMU[i][19]

#meanCadence
meanCadence=np.zeros((163,1))
for i in range(163):
    meanCadence[i][0] = V_IMU[i][17]

#meanPitchToeOff----------脚尖离地俯仰角的均值
meanPitchToeOff=np.zeros((163,1))
for i in range(163):
    meanPitchToeOff[i][0]=V_IMU[i][13]

#meanPathLength3D
meanPathLength3D=np.zeros((163,1))
for i in range(163):
    meanPathLength3D[i][0]=V_IMU[i][23]

#meanPathLength2D
meanPathLength2D=np.zeros((163,1))
for i in range(163):
    meanPathLength2D[i][0]=V_IMU[i][25]



#CVStride_Velocity
CVStride_Velocity=np.zeros((163,1))
for i in range(163):
    CVStride_Velocity[i][0]=(V_IMU[i][22]/V_IMU[i][4])/(V_IMU[i][21]/V_IMU[i][3])*100

#CVClearance
CVClearance=np.zeros((163,1))
for i in range(163):
    CVClearance[i][0] = V_IMU[i][28]/V_IMU[i][27]*100

#CV%Loading
CVLoading = np.zeros((163, 1))
for i in range(163):
    CVLoading[i][0] = V_IMU[i][6]/V_IMU[i][5]*100

#CV%Pushing
CVPushing=np.zeros((163,1))
for i in range(163):
    CVPushing[i][0]=V_IMU[i][10]/V_IMU[i][9]*100

#CV%Swing
CVSwing = np.zeros((163, 1))
for i in range(163):
    CVSwing[i][0] = V_IMU[i][12]/V_IMU[i][11]*100

#CV%FootFlat----------落地后，把身体重量慢慢压到脚上
CVFootFlat=np.zeros((163,1))
for i in range(163):
    CVFootFlat[i][0] = V_IMU[i][8]/V_IMU[i][7]*100

#CVPitchHS----------Heel-Strike脚后跟着地俯仰角的变异系数
CVPitchHS=np.zeros((163,1))
for i in range(163):
    CVPitchHS[i][0]=V_IMU[i][16]/V_IMU[i][15]*100

#CVStride_Length-------变异系数CV=标准差与均值的比值
CVStride_Length=np.zeros((163,1))
for i in range(163):
    CVStride_Length[i][0]=V_IMU[i][22]/V_IMU[i][21]*100

#CVStep_Speed
CVStep_Speed=np.zeros((163,1))
for i in range(163):
    CVStep_Speed[i][0]=V_IMU[i][20]/V_IMU[i][19]*100

#CVCadence
CVCadence=np.zeros((163,1))
for i in range(163):
    CVCadence[i][0] = V_IMU[i][18]/V_IMU[i][17]*100

#CVPitchToeOff----------脚尖离地俯仰角的变异系数------------数据变差
CVPitchToeOff=np.zeros((163,1))
for i in range(163):
    CVPitchToeOff[i][0]=V_IMU[i][14]/V_IMU[i][13]*100

#CVPathLength3D
CVPathLength3D=np.zeros((163,1))
for i in range(163):
    CVPathLength3D[i][0]=V_IMU[i][24]/V_IMU[i][23]*100

#CVPathLength2D
CVPathLength2D=np.zeros((163,1))
for i in range(163):
    CVPathLength2D[i][0]=V_IMU[i][26]/V_IMU[i][25]*100


Gait_Parameters = np.column_stack(( Distance,               Strides,                meanStride_Velocity,
                                    meanClearance,          meanLoading,            meanPushing,
                                    meanSwing,              meanFootFlat,           meanPitchHS,
                                    meanStride_Length,      meanStep_Speed,         meanCadence,
                                    meanPitchToeOff,        meanPathLength3D,       meanPathLength2D,
                                    CVStride_Velocity,      CVClearance,            CVLoading,
                                    CVPushing,              CVSwing,                CVFootFlat,
                                    CVPitchHS,              CVStride_Length,        CVStep_Speed,
                                    CVCadence,              CVPitchToeOff,          CVPathLength3D,
                                    CVPathLength2D))

Gait_Parameters_name =             ["Distance",             "Strides",              "meanStride_Velocity",
                                    "meanClearance",        "meanLoading",          "meanPushing",
                                    "meanSwing",            "meanFootFlat",         "meanPitchHS",
                                    "meanStride_Length",    "meanStep_Speed",       "meanCadence",
                                    "meanPitchToeOff",      "meanPathLength3D",     "meanPathLength2D",
                                    "CVStride_Velocity",    "CVClearance",          "CVLoading",
                                    "CVPushing",            "CVSwing",              "CVFootFlat",
                                    "CVPitchHS",            "CVStride_Length",      "CVStep_Speed",
                                    "CVCadence",            "CVPitchToeOff",        "CVPathLength3D",
                                    "CVPathLength2D"]

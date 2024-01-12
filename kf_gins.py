from angle import Angle
import gi_engine as gi
import numpy as np
import pandas as pd
import kf_gins_types as ky
import types_my as ty
import sys
import yaml

def LoadOptions():
    ## 读取初始位置(纬度 经度 高程)、(北向速度 东向速度 垂向速度)、姿态(欧拉角，ZYX旋转顺序, 横滚角、俯仰角、航向角)
    options = ky.GINSOptions()
    options.initstate.pos = np.array(config['initpos']) * Angle.D2R
    options.initstate.vel = np.array(config['initvel'])
    options.initstate.euler = np.array(config['initatt']) * Angle.D2R
    options.initstate.pos[2] *= Angle.R2D

    ## 读取IMU误差初始值(零偏和比例因子)
    options.initstate.imuerror.gyrbias = np.array(config['initgyrbias']) * Angle.D2R/3600.0
    options.initstate.imuerror.accbias = np.array(config['initaccbias']) * 1e-5
    options.initstate.imuerror.gyrscale = np.array(config['initgyrscale']) * 1e-6
    options.initstate.imuerror.accscale = np.array(config['initaccscale']) * 1e-6

    ## 读取初始位置、速度、姿态(欧拉角)的标准差
    options.initstate_std.pos = np.array(config['initposstd'])
    options.initstate_std.vel = np.array(config['initvelstd'])
    options.initstate_std.euler = np.array(config['initattstd']) * Angle.D2R

    ## 读取IMU噪声参数
    options.imunoise.gyr_arw = np.array(config['arw'])
    options.imunoise.acc_vrw = np.array(config['vrw'])
    options.imunoise.gyrbias_std = np.array(config['gbstd'])
    options.imunoise.accbias_std = np.array(config['abstd'])
    options.imunoise.gyrscale_std = np.array(config['gsstd'])
    options.imunoise.accscale_std = np.array(config['asstd'])
    options.rssinoise.rss_std = config['rsstd']
    options.imunoise.corr_time = config['corrtime']

    ## 读取IMU误差初始标准差,如果配置文件中没有设置，则采用IMU噪声参数中的零偏和比例因子的标准差
    options.initstate_std.imuerror.gyrbias = np.array(config['gbstd']) * Angle.D2R / 3600.0
    options.initstate_std.imuerror.accbias = np.array(config['abstd']) * 1e-5
    options.initstate_std.imuerror.gyrscale = np.array(config['gsstd']) * 1e-6
    options.initstate_std.imuerror.accscale = np.array(config['asstd']) * 1e-6

    ## IMU噪声参数转换为标准单位
    options.imunoise.gyr_arw *= (Angle.D2R / 60.0)
    options.imunoise.acc_vrw /= 60.0
    options.imunoise.gyrbias_std *= (Angle.D2R / 3600.0)
    options.imunoise.accbias_std *= 1e-5
    options.imunoise.gyrscale_std *= 1e-6
    options.imunoise.accscale_std *= 1e-6
    options.imunoise.corr_time *= 3600

    ## GNSS天线杆臂, GNSS天线相位中心在IMU坐标系下位置
    options.antlever_G = np.array(config['antlever_G'])
    options.antlever_B = np.array(config['antlever_B'])

    ## BLE传播参数
    options.BLE_A = config['BLE_A']
    options.BLE_n = config['BLE_n']

    ## 滤波算法
    options.filter = config['filter']

    ## 初始时间
    options.starttime = config['starttime']

    ## NHC与高度更新
    options.ifNHC = config['NHC']
    options.ifALT = config['ALT']
    options.ifHuber = config['Huber']
    options.ifChi = config['Chi']
    
    return options

def imuload(data_,rate,pre_time):
    dt_ = 1.0 / rate
    imu_ = ty.IMU()
    imu_.time = data_[0]
    imu_.dtheta = np.array(data_[1:4])
    imu_.dvel = np.array(data_[4:7])
    dt = imu_.time - pre_time
    pre_time = imu_.time
    if dt < 0.1:
        imu_.dt = dt
    else:
        imu_.dt = dt_
    return imu_,pre_time

def gnssload(data_):
    gnss_ = ty.GNSS()
    gnss_.time = data_[0]
    gnss_.blh = np.array(data_[1:4])
    gnss_.std = np.array(data_[4:7])
    gnss_.blh[0] *= Angle.D2R
    gnss_.blh[1] *= Angle.D2R
    return gnss_

def bleload(data_):
    ble_ = ty.BLE()
    ble_.time = data_['t']
    ble_.AP = len(data_['rssi'])
    ble_.RSSI = np.array(data_['rssi']).astype(float)
    ble_.blh = np.array(data_['coord'])
    ble_.blh[:,0:2] *= Angle.D2R
    ble_.alt = data_['alt']
    return ble_

def align(imu_data,gnss_data,ble_data,starttime):
    imu_cur = ty.IMU()
    gnss = ty.GNSS()
    imu_index = 0
    gnss_index = 0
    pre_time = starttime
    p_t = 0
    for index,row in enumerate(imu_data):
        imu_cur,p_t = imuload(row,imudatarate,pre_time)
        imu_index = index
        if row[0] > starttime:
            break  
    for index,row in enumerate(gnss_data):
        gnss = gnssload(row)
        gnss_index = index
        if row[0] > starttime:
            break
    for index,row in ble_data.iterrows():
        ble = bleload(row)
        ble_index = index
        if row['t'] > starttime:
            break

    return imu_cur,gnss,ble,imu_index,gnss_index,ble_index,p_t

## 数据读取
with open('kf-gins.yaml', 'r',encoding='utf-8') as file:
        config = yaml.safe_load(file)

## 初始化
nav_result = np.empty((0, 10))
error_result = np.empty((0, 14))
std_result = np.empty((0, 23))
options = LoadOptions()
giengine = gi.GIEngine()
giengine.GIFunction(options)
imudatarate = config['imudatarate']
starttime = config['starttime']
endtime = config['endtime']
pre_time = starttime

## 读取数据
imu_data = np.genfromtxt(config['imupath'],delimiter=',')
gnss_data = np.genfromtxt(config['gnsspath'],delimiter=',')
ble_data = pd.read_csv(config['blepath'], header=None, names=['t','rssi','coord','id','alt'])
ble_data['rssi'] = ble_data['rssi'].apply(lambda x: eval(x))
ble_data['coord'] = ble_data['coord'].apply(lambda x: eval(x))
ble_data['id'] = ble_data['id'].apply(lambda x: eval(x))


## 停止时间为-1时设置停止时间为数据集的最后一个时间
if endtime < 0 :
    endtime = imu_data[-1, 0]

## 初始数据对齐
imu_cur,gnss,ble,is_index,gs_index,bs_index,pre_time = align(imu_data,gnss_data,ble_data,starttime)

## 初始数据载入
giengine.addImuData(imu_cur, True)
giengine.addGnssData(gnss)
giengine.addBleData(ble)

## 循环处理数据进行定位
for row in imu_data[is_index+1:]:
    ## GNSS数据载入
    if gnss.time < imu_cur.time and gnss.time+1 <= endtime:
        gnss = gnssload(gnss_data[gs_index])
        gs_index += 1
        giengine.addGnssData(gnss)
    ## BLE数据载入
    if bs_index<ble_data.shape[0]:
        if ble.time < imu_cur.time and ble_data['t'][bs_index] < endtime:
            ble = bleload(ble_data.iloc[bs_index])
            bs_index += 1
            giengine.addBleData(ble)
    ## IMU数据载入
    imu_cur,pre_time = imuload(row,imudatarate,pre_time)
    if imu_cur.time > endtime:
        break
    giengine.addImuData(imu_cur)
    ## 处理开始
    giengine.newImuProcess()
    ## 数据输出
    timestamp = giengine.timestamp()
    navstate  = giengine.getNavState()
    imuerr = navstate.imuerror
    rssierr = navstate.rssierror
    cov       = giengine.getCovariance()
    result1 = np.array([np.round(timestamp,9),np.round(navstate.pos[1]* Angle.R2D,9),np.round(navstate.pos[0]* Angle.R2D,9),  np.round(navstate.pos[2],9),np.round(navstate.vel[0],9), np.round(navstate.vel[1],9), np.round(navstate.vel[2],9),np.round(navstate.euler[0]* Angle.R2D,9),np.round(navstate.euler[1]* Angle.R2D,9),np.round(navstate.euler[2]* Angle.R2D,9)])
    result2 = np.array([np.round(timestamp,9),np.round(imuerr.gyrbias[0]* Angle.R2D*3600,9),np.round(imuerr.gyrbias[1]* Angle.R2D*3600,9),  np.round(imuerr.gyrbias[2]* Angle.R2D*3600,9),np.round(imuerr.accbias[0]* 1e5,9), np.round(imuerr.accbias[1]* 1e5,9), np.round(imuerr.accbias[2]* 1e5,9),np.round(imuerr.gyrscale[0] * 1e6,9),np.round(imuerr.gyrscale[1] * 1e6,9),np.round(imuerr.gyrscale[2] * 1e6,9),np.round(imuerr.accscale[0] * 1e6,9),np.round(imuerr.accscale[1] * 1e6,9),np.round(imuerr.accscale[2] * 1e6,9),np.round(rssierr.brss,1)])
    std = np.sqrt(cov.diagonal())
    std[6:9] *= Angle.R2D
    std[9:12] *= Angle.R2D*3600
    std[12:15] *= 1e5
    std[15:21] *= 1e6
    std = np.round(std,6)
    result3 =np.insert(std,0,np.round(timestamp,9))
    nav_result = np.vstack((nav_result, result1))
    error_result = np.vstack((error_result, result2))
    std_result = np.vstack((std_result, result3))
    sys.stdout.write('\r' + str(timestamp))
    sys.stdout.flush()

## 保存数据
np.savetxt(config['outputpath_nav'], nav_result, delimiter=",",fmt="%6f")    
np.savetxt(config['outputpath_error'], error_result, delimiter=",",fmt="%6f") 
np.savetxt(config['outputpath_std'], std_result, delimiter=",",fmt="%6f") 





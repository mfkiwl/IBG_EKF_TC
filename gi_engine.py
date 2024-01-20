import types_my as ty
import kf_gins_types as kf
from angle import Angle
from rotation import Rotation_my as ro
from insmech import INSMech as INS
from earth import Earth
from enum import IntEnum
import numpy as np 
import os
from scipy.spatial.transform import Rotation


class StateID(IntEnum):
    P_ID = 0
    V_ID = 3
    PHI_ID = 6
    BG_ID = 9
    BA_ID = 12
    SG_ID = 15
    SA_ID = 18
    BRSS_ID = 21

class NoiseID(IntEnum):
    VRW_ID = 0
    ARW_ID = 3
    BGSTD_ID = 6
    BASTD_ID = 9
    SGSTD_ID = 12
    SASTD_ID = 15
    BRSTD_ID = 18
    
class GIEngine:
    
    ##  更新时间对齐误差，IMU状态和观测信息误差小于它则认为两者对齐
    TIME_ALIGN_ERR = 0.001
    ## Kalman滤波相关
    RANK = 22
    NOISERANK = 19

    def __init__(self):
        self.options_ = kf.GINSOptions()
        self.timestamp_ = 0.0
        ##   IMU和BLE\GNSS原始数据
        self.imupre_ = ty.IMU()
        self.imucur_ = ty.IMU()
        self.bledata_ = ty.BLE()
        self.gnssdata_ = ty.GNSS()
        ##  IMU状态（位置、速度、姿态和IMU误差）
        self.pvacur_ = kf.PVA()
        self.pvapre_ = kf.PVA()
        self.imuerror_ = kf.ImuError()
        ##  RSSI误差状态
        self.rssierro_  = kf.RSSIError()
        ## Kalman滤波相关
        self.Cov_ = np.zeros((GIEngine.RANK,GIEngine.RANK))
        self.Qc_ = np.zeros((GIEngine.NOISERANK,GIEngine.NOISERANK))
        self.dx_ = np.zeros((GIEngine.RANK,1))
        self.transformed_deviation = np.zeros((GIEngine.RANK, 2*GIEngine.RANK+1))
        ## NHC相关
        self.NHC_data = kf.NHCData()
    
    def GIFunction(self,options:kf.GINSOptions):
        self.options_ = options
        self.timestamp_ = 0.0
        imunoise  = self.options_.imunoise
        rssinoise = self.options_.rssinoise
        ## 初始化系统噪声阵
        self.Qc_[NoiseID.ARW_ID:NoiseID.ARW_ID+3, NoiseID.ARW_ID:NoiseID.ARW_ID+3] = np.diag(np.square(imunoise.gyr_arw))
        self.Qc_[NoiseID.VRW_ID:NoiseID.VRW_ID+3, NoiseID.VRW_ID:NoiseID.VRW_ID+3] = np.diag(np.square(imunoise.acc_vrw))
        self.Qc_[NoiseID.BGSTD_ID:NoiseID.BGSTD_ID+3, NoiseID.BGSTD_ID:NoiseID.BGSTD_ID+3] = 2/imunoise.corr_time * np.diag(np.square(imunoise.gyrbias_std))
        self.Qc_[NoiseID.BASTD_ID:NoiseID.BASTD_ID+3, NoiseID.BASTD_ID:NoiseID.BASTD_ID+3] = 2/imunoise.corr_time * np.diag(np.square(imunoise.accbias_std))
        self.Qc_[NoiseID.SGSTD_ID:NoiseID.SGSTD_ID+3, NoiseID.SGSTD_ID:NoiseID.SGSTD_ID+3] = 2/imunoise.corr_time * np.diag(np.square(imunoise.gyrscale_std))
        self.Qc_[NoiseID.SASTD_ID:NoiseID.SASTD_ID+3, NoiseID.SASTD_ID:NoiseID.SASTD_ID+3] = 2/imunoise.corr_time * np.diag(np.square(imunoise.accscale_std))
        # self.Qc_[NoiseID.BRSTD_ID, NoiseID.BRSTD_ID] = 2/imunoise.corr_time*np.square(rssinoise.rss_std)
        self.Qc_[NoiseID.BRSTD_ID, NoiseID.BRSTD_ID] = np.square(rssinoise.rss_std)
        ## 设置系统状态(位置、速度、姿态和IMU误差)初值和初始协方差
        initstate = self.options_.initstate
        initstate_std = self.options_.initstate_std
        self.initialize(initstate, initstate_std)
        ##设置NHC初始时间
        self.NHC_data.time = self.options_.starttime
        
    def initialize(self,initstate:kf.NavState,initstate_std:kf.NavState):
        # 初始化位置、速度、姿态
        self.pvacur_.pos= initstate.pos
        self.pvacur_.vel= initstate.vel
        self.pvacur_.att.euler= initstate.euler
        self.pvacur_.att.cbn = ro.euler2matrix( initstate.euler)
        self.pvacur_.att.qbn = ro.euler2quaternion( np.flip(initstate.euler, axis=0))
        # 初始化IMU误差
        self.imuerror_ = initstate.imuerror
        # 初始化RSSI误差
        self.rssierro_ = initstate.rssierror
        # 给上一时刻状态赋同样的初值
        p = self.pvacur_
        self.pvapre_ = p
        # 初始化协方差
        imuerror_std = initstate_std.imuerror
        rssierror_std = initstate_std.rssierror
        rssierror_std.brss = 0.0000000000000000000000000000001
        self.Cov_[StateID.P_ID:StateID.P_ID+3,StateID.P_ID:StateID.P_ID+3]= \
        np.diag(np.square(initstate_std.pos))
        self.Cov_[StateID.V_ID:StateID.V_ID+3,StateID.V_ID:StateID.V_ID+3]= \
        np.diag(np.square(initstate_std.vel))
        self.Cov_[StateID.PHI_ID:StateID.PHI_ID+3,StateID.PHI_ID:StateID.PHI_ID+3]= \
        np.diag(np.square(initstate_std.euler))
        self.Cov_[StateID.BG_ID:StateID.BG_ID+3,StateID.BG_ID:StateID.BG_ID+3]= \
        np.diag(np.square(imuerror_std.gyrbias))
        self.Cov_[StateID.BA_ID:StateID.BA_ID+3,StateID.BA_ID:StateID.BA_ID+3]= \
        np.diag(np.square(imuerror_std.accbias))
        self.Cov_[StateID.SG_ID:StateID.SG_ID+3,StateID.SG_ID:StateID.SG_ID+3]= \
        np.diag(np.square(imuerror_std.gyrscale))
        self.Cov_[StateID.SA_ID:StateID.SA_ID+3,StateID.SA_ID:StateID.SA_ID+3]= \
        np.diag(np.square(imuerror_std.accscale))
        self.Cov_[StateID.BRSS_ID,StateID.BRSS_ID]= \
        np.square(rssierror_std.brss)

    def newImuProcess(self):
        ## 当前IMU时间作为系统当前状态时间
        time = self.imucur_.time
        self.timestamp_ = time

        ## NHC更新
        if self.options_.ifNHC == 1:
            self.NHC_judge()

        if self.timestamp_ <=357514 or 357561>=self.timestamp_>=357549:
            self.Cov_[21,21] = 1

        ## 如果GNSS有效，则将GNSS更新时间设置为GNSS时间
        if self.gnssdata_.isvalid:
            updatetime_G = self.gnssdata_.time
        else:
            updatetime_G = -1
        ## 如果BLE有效，则将BLE更新时间设置为BLE时间
        if self.bledata_.isvalid:
            updatetime_B = self.bledata_.time
        else:
            updatetime_B = -1
        
        imupre_ = self.imupre_
        imucur_ = self.imucur_
        gnssdata_ = self.gnssdata_
        bledata_ = self.bledata_

        ## 判断是否需要进行GNSS更新
        res_G = self.GisToUpdate(imupre_.time, imucur_.time, updatetime_G)
        ## 判断是否要进行BLE更新
        res_B = self.BisToUpdate(imupre_.time, imucur_.time, updatetime_B)
    
        if res_B ==0:
            if res_G == 0:
            ## 只传播导航状态
                self.insPropagation(imupre_, imucur_)
            elif res_G == 1:
                ## GNSS数据靠近上一历元，先对上一历元进行GNSS更新
                self.gnssUpdate(gnssdata_,self.options_.filter)
                self.stateFeedback()
                self.pvapre_ = self.pvacur_
                self.insPropagation(imupre_, imucur_)
            elif res_G == 2:
                ## GNSS数据靠近当前历元，先对当前IMU进行状态传播
                self.insPropagation(imupre_, imucur_)
                self.gnssUpdate(gnssdata_,self.options_.filter)
                self.stateFeedback()
            else:
                ## GNSS数据在两个IMU数据之间(不靠近任何一个), 将当前IMU内插到整秒时刻
                midimu = ty.IMU
                imucur_, midimu = GIEngine.imuInterpolate(imupre_, imucur_, updatetime_G, midimu)
                ## 对前一半IMU进行状态传播
                self.insPropagation(imupre_, midimu)
                ## 整秒时刻进行GNSS更新，并反馈系统状态
                self.gnssUpdate(gnssdata_,self.options_.filter)
                self.stateFeedback()
                ## 对后一半IMU进行状态传播
                self.pvapre_ = self.pvacur_
                self.insPropagation(midimu, imucur_)
        else:
            if res_G == 0:
                ## 进入室内后使用UKF
                self.bleUpdate(bledata_,self.options_.filter)
                self.stateFeedback()
                self.pvapre_ = self.pvacur_
                self.insPropagation(imupre_, imucur_)
            elif res_G == 1:
                ## GNSS数据靠近上一历元，先对上一历元进行GNSS更新
                self.ble_gnssUpdate(gnssdata_,bledata_,self.options_.filter)
                self.stateFeedback()
                self.pvapre_ = self.pvacur_
                self.insPropagation(imupre_, imucur_)
            elif res_G == 2:
                ## GNSS数据靠近当前历元，先对当前IMU进行状态传播
                self.insPropagation(imupre_, imucur_)
                self.ble_gnssUpdate(gnssdata_,bledata_,self.options_.filter)
                self.stateFeedback()
            else:
                ## GNSS数据在两个IMU数据之间(不靠近任何一个), 将当前IMU内插到整秒时刻
                midimu = ty.IMU
                imucur_, midimu = GIEngine.imuInterpolate(imupre_, imucur_, updatetime_G, midimu)
                ## 对前一半IMU进行状态传播
                self.insPropagation(imupre_, midimu)
                ## 整秒时刻进行GNSS更新，并反馈系统状态
                self.ble_gnssUpdate(gnssdata_,bledata_,self.options_.filter)
                self.stateFeedback()
                ## 对后一半IMU进行状态传播
                self.pvapre_ = self.pvacur_
                self.insPropagation(midimu, imucur_)
            if self.options_.ifALT == 1:
                self.altitude_Update(bledata_)
                self.stateFeedback()

        ## 检查协方差矩阵对角线元素
        self.checkCov()

        ## 更新上一时刻的状态和IMU数据
        self.pvapre_ = self.pvacur_
        self.imupre_ = self.imucur_

    def checkCov(self):
        for i in range(GIEngine.RANK):
            if self.Cov_[i, i] < 0:
                print(f"Covariance is negative at {self.timestamp_:.10f} !")
                exit(os.EXIT_FAILURE)

    def imuCompensate(self,imu:ty.IMU) -> ty.IMU :
        ## 补偿IMU零偏误差
        imu.dtheta -= self.imuerror_.gyrbias * imu.dt
        imu.dvel -= self.imuerror_.accbias * imu.dt
        ## 补偿IMU比例因子误差
        one = np.ones(3)
        gyrscale = one + self.imuerror_.gyrscale
        accscale = one + self.imuerror_.accscale
        imu.dtheta = np.multiply(imu.dtheta,np.reciprocal(gyrscale))
        imu.dvel = np.multiply(imu.dvel,np.reciprocal(accscale))
        return imu
    
    def insPropagation(self,imupre:ty.IMU,imucur:ty.IMU):
        ## 对当前IMU数据(imucur)补偿误差, 上一IMU数据(imupre)已经补偿过了
        imucur = self.imuCompensate(imucur)
        ## IMU状态更新(机械编排算法)
        pvapre_ = self.pvapre_
        self.pvacur_ = INS.insMech(pvapre_, imupre, imucur)
        
        ## 系统噪声传播，姿态误差采用phi角误差模型
        ## 初始化Phi阵(状态转移矩阵)，F阵，Qd阵(传播噪声阵)，G阵(噪声驱动阵)
        # Phi = np.identity(self.Cov_.shape[0])
        # F = np.zeros_like(self.Cov_)
        # Qd = np.zeros_like(self.Cov_)
        Phi = np.identity(22)
        F = np.zeros((22,22))
        Qd = np.zeros((22,22))
        G = np.zeros((GIEngine.RANK, GIEngine.NOISERANK))
        ## 使用上一历元状态计算状态转移矩阵
        rmrn = Earth.meridianPrimeVerticalRadius(pvapre_.pos[0])
        gravity = Earth.gravity(pvapre_.pos)
        wie_n = np.array([Earth.WGS84_WIE * np.cos(pvapre_.pos[0]), 0, -Earth.WGS84_WIE * np.sin(pvapre_.pos[0])])
        wen_n = np.array([pvapre_.vel[1] / (rmrn[1] + pvapre_.pos[2]), -pvapre_.vel[0] / (rmrn[0] + pvapre_.pos[2]),-pvapre_.vel[1] * np.tan(pvapre_.pos[0]) / (rmrn[1] + pvapre_.pos[2])])
        rmh   = rmrn[0] + pvapre_.pos[2]
        rnh   = rmrn[1] + pvapre_.pos[2]
        accel = imucur.dvel / imucur.dt
        omega = imucur.dtheta / imucur.dt

        ## 位置误差
        temp = np.zeros((3,3))
        temp[0,0] = -pvapre_.vel[2] / rmh
        temp[0,2] =  pvapre_.vel[0] / rmh
        temp[1,0] = pvapre_.vel[1] * np.tan(pvapre_.pos[0]) / rnh
        temp[1,1] = -(pvapre_.vel[2] + pvapre_.vel[0] * np.tan(pvapre_.pos[0])) / rnh
        temp[1,2] = pvapre_.vel[1] / rnh
        F[StateID.P_ID:StateID.P_ID+3,StateID.P_ID:StateID.P_ID+3] = temp
        F[StateID.P_ID:StateID.P_ID+3,StateID.V_ID:StateID.V_ID+3] = np.identity(3)

        ## 速度误差
        temp = np.zeros((3,3))
        temp[0, 0] = -2 * pvapre_.vel[1] * Earth.WGS84_WIE * np.cos(pvapre_.pos[0]) / rmh -np.power(pvapre_.vel[1], 2) / rmh / rnh / np.power(np.cos(pvapre_.pos[0]), 2)
        temp[0, 2] = pvapre_.vel[0] * pvapre_.vel[2] / rmh / rmh - np.power(pvapre_.vel[1], 2) * np.tan(pvapre_.pos[0]) / rnh / rnh
        temp[1, 0] = 2 * Earth.WGS84_WIE * (pvapre_.vel[0] * np.cos(pvapre_.pos[0]) - pvapre_.vel[2] * np.sin(pvapre_.pos[0])) / rmh + pvapre_.vel[0] * pvapre_.vel[1] / rmh / rnh / np.power(np.cos(pvapre_.pos[0]), 2)
        temp[1, 2] = (pvapre_.vel[1] * pvapre_.vel[2] + pvapre_.vel[0] * pvapre_.vel[1] * np.tan(pvapre_.pos[0])) / rnh / rnh
        temp[2, 0] = 2 * Earth.WGS84_WIE * pvapre_.vel[1] * np.sin(pvapre_.pos[0]) / rmh
        temp[2, 2] = -np.power(pvapre_.vel[1], 2) / rnh / rnh - np.power(pvapre_.vel[0], 2) / rmh / rmh +2 * gravity / (np.sqrt(rmrn[0] * rmrn[1]) + pvapre_.pos[2])
        F[StateID.V_ID:StateID.V_ID+3,StateID.P_ID:StateID.P_ID+3] = temp
        temp = np.zeros((3,3))
        temp[0, 0] = pvapre_.vel[2] / rmh
        temp[0, 1] = -2 * (Earth.WGS84_WIE * np.sin(pvapre_.pos[0]) + pvapre_.vel[1] * np.tan(pvapre_.pos[0]) / rnh)
        temp[0, 2] = pvapre_.vel[0] / rmh
        temp[1, 0] = 2 * Earth.WGS84_WIE * np.sin(pvapre_.pos[0]) + pvapre_.vel[1] * np.tan(pvapre_.pos[0]) / rnh
        temp[1, 1] = (pvapre_.vel[2] + pvapre_.vel[0] * np.tan(pvapre_.pos[0])) / rnh
        temp[1, 2] = 2 * Earth.WGS84_WIE * np.cos(pvapre_.pos[0]) + pvapre_.vel[1] / rnh
        temp[2, 0] = -2 * pvapre_.vel[0] / rmh
        temp[2, 1] = -2 * (Earth.WGS84_WIE * np.cos(pvapre_.pos[0]) + pvapre_.vel[1] / rnh)
        F[StateID.V_ID:StateID.V_ID+3,StateID.V_ID:StateID.V_ID+3] = temp
        F[StateID.V_ID:StateID.V_ID+3,StateID.PHI_ID:StateID.PHI_ID+3] = ro.skewSymmetric(pvapre_.att.cbn @ accel)
        F[StateID.V_ID:StateID.V_ID+3,StateID.BA_ID:StateID.BA_ID+3] = pvapre_.att.cbn
        F[StateID.V_ID:StateID.V_ID+3,StateID.SA_ID:StateID.SA_ID+3] = pvapre_.att.cbn @ np.diag(accel)

        ## 姿态误差
        temp = np.zeros((3,3))
        temp[0, 0] = -Earth.WGS84_WIE * np.sin(pvapre_.pos[0]) / rmh
        temp[0, 2] = pvapre_.vel[1] / rnh / rnh
        temp[1, 2] = -pvapre_.vel[0] / rmh / rmh
        temp[2, 0] = -Earth.WGS84_WIE * np.cos(pvapre_.pos[0]) / rmh - pvapre_.vel[1] / rmh / rnh / np.power(np.cos(pvapre_.pos[0]), 2)
        temp[2, 2] = -pvapre_.vel[1] * np.tan(pvapre_.pos[0]) / rnh / rnh
        F[StateID.PHI_ID:StateID.PHI_ID+3,StateID.P_ID:StateID.P_ID+3] = temp
        temp = np.zeros((3,3))
        temp[0, 1] = 1 / rnh
        temp[1, 0] = -1 / rmh
        temp[2, 1] = -np.tan(pvapre_.pos[0]) / rnh
        F[StateID.PHI_ID:StateID.PHI_ID+3,StateID.V_ID:StateID.V_ID+3] = temp
        F[StateID.PHI_ID:StateID.PHI_ID+3,StateID.PHI_ID:StateID.PHI_ID+3] = -ro.skewSymmetric(wie_n + wen_n)
        F[StateID.PHI_ID:StateID.PHI_ID+3,StateID.BG_ID:StateID.BG_ID+3] = -pvapre_.att.cbn
        F[StateID.PHI_ID:StateID.PHI_ID+3,StateID.SG_ID:StateID.SG_ID+3] = -pvapre_.att.cbn @ np.diag(omega)

        ## IMU零偏误差和比例因子误差，建模成一阶高斯-马尔科夫过程
        F[StateID.BG_ID:StateID.BG_ID+3,StateID.BG_ID:StateID.BG_ID+3] = -1 / self.options_.imunoise.corr_time * np.identity(3)
        F[StateID.BA_ID:StateID.BA_ID+3,StateID.BA_ID:StateID.BA_ID+3] = -1 / self.options_.imunoise.corr_time * np.identity(3)
        F[StateID.SG_ID:StateID.SG_ID+3,StateID.SG_ID:StateID.SG_ID+3] = -1 / self.options_.imunoise.corr_time * np.identity(3)
        F[StateID.SA_ID:StateID.SA_ID+3,StateID.SA_ID:StateID.SA_ID+3] = -1 / self.options_.imunoise.corr_time * np.identity(3)

        ## RSSI误差
        # F[StateID.BRSS_ID,StateID.BRSS_ID] = -1 / self.options_.imunoise.corr_time
        F[StateID.BRSS_ID,StateID.BRSS_ID] = 0.0

        ## 系统噪声驱动矩阵
        G[StateID.V_ID:StateID.V_ID+3,NoiseID.VRW_ID:NoiseID.VRW_ID+3] =  pvapre_.att.cbn
        G[StateID.PHI_ID:StateID.PHI_ID+3,NoiseID.ARW_ID:NoiseID.ARW_ID+3] =  pvapre_.att.cbn
        G[StateID.BG_ID:StateID.BG_ID+3,NoiseID.BGSTD_ID:NoiseID.BGSTD_ID+3] = np.identity(3)
        G[StateID.BA_ID:StateID.BA_ID+3,NoiseID.BASTD_ID:NoiseID.BASTD_ID+3] = np.identity(3)
        G[StateID.SG_ID:StateID.SG_ID+3,NoiseID.SGSTD_ID:NoiseID.SGSTD_ID+3] = np.identity(3)
        G[StateID.SA_ID:StateID.SA_ID+3,NoiseID.SASTD_ID:NoiseID.SASTD_ID+3] = np.identity(3)
        G[StateID.BRSS_ID,NoiseID.BRSTD_ID] = 1.0

        ## 状态转移矩阵
        Phi = Phi + F * imucur.dt
        ## 计算系统传播噪声
        Qd = G @ self.Qc_ @ G.T * imucur.dt
        
        Qd = (Phi @ Qd @ Phi.T + Qd) / 2
        
        ## EKF预测传播系统协方差和系统误差状态
        if self.options_.filter == 'EKF':
            self.EKFPredict(Phi, Qd)
        elif self.options_.filter == 'UKF':
            self.UKFPredict(Phi, Qd)

    def GisToUpdate(self,imutime1:float,imutime2:float,updatetime_G:float) -> int:
        if np.abs(imutime1 - updatetime_G) < GIEngine.TIME_ALIGN_ERR :
            ## 更新时间靠近imutime1
            return 1
        elif np.abs(imutime2 - updatetime_G) <= GIEngine.TIME_ALIGN_ERR :
            ## 更新时间靠近imutime2
            return 2
        elif imutime1 < updatetime_G and updatetime_G < imutime2 :
            ## 更新时间在imutime1和imutime2之间, 但不靠近任何一个
            return 3
        else:
            ## 更新时间不在imutimt1和imutime2之间，且不靠近任何一个
            return 0
        
    def BisToUpdate(self,imutime1:float,imutime2:float,updatetime_B:float) -> int:
        if np.abs(imutime1 - updatetime_B) < GIEngine.TIME_ALIGN_ERR :
            ## 更新时间靠近imutime1
            return 1
        elif np.abs(imutime2 - updatetime_B) <= GIEngine.TIME_ALIGN_ERR :
            ## 更新时间靠近imutime2
            return 2
        elif imutime1 < updatetime_B and updatetime_B < imutime2 :
            ## 更新时间在imutime1和imutime2之间, 但不靠近任何一个
            return 3
        else:
            ## 更新时间不在imutimt1和imutime2之间，且不靠近任何一个
            return 0
        
    def EKFPredict(self,Phi:np.ndarray,Qd:np.ndarray):
        assert Phi.shape[0] == self.Cov_.shape[0]
        assert Qd.shape[0] == self.Cov_.shape[0]
        
        ## 传播系统协方差和误差状态
        self.Cov_ = Phi @ self.Cov_ @ Phi.transpose() + Qd
        self.dx_  = Phi @ self.dx_
        
    def EKFUpdate(self,dz:np.ndarray,H:np.ndarray,R:np.ndarray):
        assert  H.shape[1] == self.Cov_.shape[0]
        assert  dz.shape[0] == H.shape[0]
        assert  dz.shape[0] == R.shape[0]
        assert  dz.shape[1] == 1

        ## 计算Kalman增益
        temp =  H @ self.Cov_ @ H.transpose() + R
        K =  self.Cov_ @ H.transpose() @ np.linalg.inv(temp)

        ## 更新系统误差状态和协方差
        I = np.identity(self.Cov_.shape[0])
        I = I - K @ H
        ## 如果每次更新后都进行状态反馈，则更新前dx_一直为0，下式可以简化为：dx_ = K * dz
        self.dx_  = self.dx_ + K @ (dz - H @ self.dx_)
        self.Cov_ = I @ self.Cov_ @ I.transpose() + K @ R @ K.transpose()
    
    def UKFPredict(self,Phi:np.ndarray,Qd:np.ndarray):
        n = GIEngine.RANK
        alpha = 1.0
        kappa = 0.0
        beta = 2.0

        lambda_ = alpha**2 * (n + kappa) - n
        Wm = np.full(2*n+1, 0.5/(n+lambda_))
        Wm[0] = lambda_/(n+lambda_)
        Wc = np.copy(Wm)
        Wc[0] += (1 - alpha**2 + beta)

        x = np.copy(self.dx_).reshape((1, n))
        Cov = np.copy(self.Cov_)

        def is_positive_definite(matrix):
            try:
                np.linalg.cholesky(matrix)
                return False
            except np.linalg.LinAlgError:
                return True
        if is_positive_definite(Cov):
            Cov[21,21] = 0.1
        
        # 生成 Sigma Points
        sqrt_P = np.linalg.cholesky(n * Cov)
        sigma_points = np.zeros((n, 2*n+1))
        sigma_points[:, 0] = x
        sigma_points[:, 1:n+1] = x.reshape((n, 1)) + sqrt_P
        sigma_points[:, n+1:] = x.reshape((n, 1)) - sqrt_P
        sigma_points_pred = np.array([(Phi @ sp) for sp in sigma_points.T]).T
        # 预测均值和协方差
        x_pred = np.dot(Wm, sigma_points_pred.T)
        self.transformed_deviation = sigma_points_pred - x_pred.reshape((n, 1))
        self.Cov_ = np.dot(Wc * self.transformed_deviation, self.transformed_deviation.T) + Qd
        self.dx_ = x_pred.reshape((n, 1))

    def UKFUpdate(self,dz:np.ndarray,H_gnss,R:np.ndarray,antenna_pos:np.ndarray,bledata:ty.BLE):
        n = GIEngine.RANK
        alpha = 1.0
        kappa = 0.0
        beta = 2.0
        ##
        lambda_ = alpha**2 * (n + kappa) - n
        Wm = np.full(2*n+1, 0.5/(n+lambda_))
        Wm[0] = lambda_/(n+lambda_)
        Wc = np.copy(Wm)
        Wc[0] += (1 - alpha**2 + beta)
        x_pred = np.copy(self.dx_).reshape((1, n))
        Cov_pred = np.copy(self.Cov_)
        ## 计算 Sigma 点集
        sqrt_Cov_pred = np.linalg.cholesky(n * Cov_pred)
        sigma_points = np.zeros((n, 2*n+1))
        sigma_points[:, 0] = x_pred
        sigma_points[:, 1:n+1] = x_pred.reshape((n, 1)) + sqrt_Cov_pred
        sigma_points[:, n+1:] = x_pred.reshape((n, 1)) - sqrt_Cov_pred
        ## 通过非线性观测方程转换 Sigma Points
        if type(H_gnss) == int :
            transformed_sigma_points = np.array([self.ble_ob_func(sp,antenna_pos,bledata) for sp in sigma_points.T]).T
        elif type(bledata) == int:
            transformed_sigma_points = np.array([(H_gnss @ sp) for sp in sigma_points.T]).T
        else:
            transformed_sigma_points = np.array([self.ble_gnss_ob_func(sp,H_gnss,antenna_pos,bledata) for sp in sigma_points.T]).T
        # 计算观测的均值和协方差
        dz_pred = np.dot(transformed_sigma_points,Wm)
        Pz_pred = np.dot(Wc * (transformed_sigma_points - dz_pred[:, np.newaxis])  , (transformed_sigma_points - dz_pred[:, np.newaxis]).T) + R
        ## 计算交叉协方差
        Pxz = np.dot( Wc * self.transformed_deviation , (transformed_sigma_points - dz_pred[:, np.newaxis]).T)
        # 更新卡尔曼增益、状态估计和协方差
        if self.options_.ifChi == 1:
            rho = self.ChiSquare(dz,dz_pred,Pz_pred)
            K = np.dot(Pxz, rho @ np.linalg.inv(Pz_pred))
            self.dx_ = (x_pred.T + np.dot(K, dz - dz_pred[:, np.newaxis]))
            self.Cov_ = self.Cov_ - np.dot(np.dot(K, Pz_pred), K.T)
        else:
            K = np.dot(Pxz, np.linalg.inv(Pz_pred))
            self.dx_ = (x_pred.T + np.dot(K, dz - dz_pred[:, np.newaxis]))
            self.Cov_ = self.Cov_ - np.dot(np.dot(K, Pz_pred), K.T)
    
    def gnssUpdate(self,gnssdata:ty.GNSS,filter):
        ## IMU位置转到GNSS天线相位中心位置
        Dr_inv = Earth.DRi(self.pvacur_.pos)
        Dr = Earth.DR(self.pvacur_.pos)
        antenna_pos = self.pvacur_.pos + Dr_inv @ self. pvacur_.att.cbn @ self.options_.antlever_G
        ## GNSS位置测量新息
        dz = Dr @ (antenna_pos - gnssdata.blh)
        ## 构造GNSS位置观测矩阵
        H_gnsspos = np.zeros((3, self.Cov_.shape[0]))
        H_gnsspos[0:3,StateID.P_ID:StateID.P_ID+3] = np.identity(3)
        H_gnsspos[0:3,StateID.PHI_ID:StateID.PHI_ID+3] = ro.skewSymmetric(self.pvacur_.att.cbn @ self.options_.antlever_G)
        ## 位置观测噪声阵
        R_gnsspos = np.diag(np.multiply(gnssdata.std, gnssdata.std))
        ## EKF更新协方差和误差状态
        dz = dz.reshape(3, 1)
        if filter == 'EKF':
            self.EKFUpdate(dz, H_gnsspos, R_gnsspos)
        elif filter == 'UKF':
            bledata = 0
            self.UKFUpdate(dz, H_gnsspos, R_gnsspos,antenna_pos,bledata)
        ## GNSS更新之后设置为不可用
        self.gnssdata_.isvalid = False

    def bleUpdate(self,bledata:ty.BLE,filter):
        ## IMU位置转到BLE天线相位中心位置
        Dr_inv = Earth.DRi(self.pvacur_.pos)
        Dr = Earth.DR(self.pvacur_.pos)
        antenna_pos = self.pvacur_.pos + Dr_inv @ self. pvacur_.att.cbn @ self.options_.antlever_B
        ## RSSI误差补偿
        bledata.RSSI -= self.rssierro_.brss
        if filter == 'EKF':
            ## 距离测量新息与H阵中的一部分Gm
            dz = np.zeros(bledata.AP)
            Gm = np.zeros((bledata.AP,3))
            Bm = np.zeros(bledata.AP)
            d = np.zeros(bledata.AP)
            for i in range(bledata.AP):
                dI = np.linalg.norm(Dr @ (antenna_pos - bledata.blh[i]))
                dB = 10**((self.options_.BLE_A - bledata.RSSI[i])/(10*self.options_.BLE_n))
                zk = dI - dB
                dz[i] = zk
                e = (1+np.log(10)*self.rssierro_.brss/10*self.options_.BLE_n)/dI * (Dr @ (antenna_pos - bledata.blh[i]))  
                Gm[i] = e
                Bm[i] = np.log(10)*dI/10*self.options_.BLE_n
                d[i] = dI
            Gm = Gm.reshape(bledata.AP,3)
            Bm = Bm.reshape(bledata.AP, 1)
            ## BLE观测矩阵
            H_ble = np.zeros((bledata.AP, self.Cov_.shape[0]))
            H_ble[0:bledata.AP,StateID.P_ID:StateID.P_ID+3] = Gm
            H_ble[0:bledata.AP,StateID.BRSS_ID:StateID.BRSS_ID+1] = Bm
            ## 位置观测噪声阵
            # d_std = np.log(10)*self.options_.rssinoise.rss_std/(10*self.options_.BLE_n)*d
            # d_std = (4/29)*d+(25/29)
            d_std = self.options_.rssinoise.rss_std +1
            # R_d = np.diag(np.multiply(d_std, d_std))
            R_d = np.diag(np.full(bledata.AP,d_std**2))
            ## EKF更新协方差和误差状态
            dz = dz.reshape(bledata.AP, 1)
            self.EKFUpdate(dz, H_ble, R_d)
        elif filter == 'UKF':
            dz = np.zeros(bledata.AP)
            d = np.zeros(bledata.AP)
            H_gnss = 0
            for i in range(bledata.AP):
                dI = np.linalg.norm(Dr @ (antenna_pos - bledata.blh[i]))
                dB = 10**((self.options_.BLE_A - bledata.RSSI[i])/(10*self.options_.BLE_n))
                zk = dI - dB
                dz[i] = zk
                d[i] = dI
            # d_std = np.log(10)*self.options_.rssinoise.rss_std/(10*self.options_.BLE_n)*d/1.5
            # d_std = np.full(bledata.AP,self.options_.rssinoise.rss_std+1)
            if self.options_.ifNHC == 0 :
                d_std = (4/29)*d+(25/29)+1
                R_d = np.diag(np.multiply(d_std, d_std)) 
            else:
                # d_std = self.options_.rssinoise.rss_std +1
                # R_d = np.diag(np.full(bledata.AP,d_std**2))
                d_std = np.log(10)*self.options_.rssinoise.rss_std/(10*self.options_.BLE_n)*d/1.5
                R_d = np.diag(np.multiply(d_std, d_std)) 
            if self.options_.ifHuber == 1:
                R_d = self.HuberM(dz,R_d,'ble',H_gnss,antenna_pos,bledata)
                # dz = self.HuberM(dz,R_d,'ble',H_gnss,antenna_pos,bledata)
            dz = dz.reshape(bledata.AP, 1)
            self.UKFUpdate(dz, H_gnss, R_d, antenna_pos,bledata)
        ## BLE更新之后设置为不可用
        self.bledata_.isvalid = False

    def ble_gnssUpdate(self,gnssdata:ty.GNSS,bledata:ty.BLE,filter):
        ## IMU位置转到BLE和GNSS天线相位中心位置
        Dr_inv = Earth.DRi(self.pvacur_.pos)
        Dr = Earth.DR(self.pvacur_.pos)
        antenna_pos_G = self.pvacur_.pos + Dr_inv @ self. pvacur_.att.cbn @ self.options_.antlever_G
        antenna_pos_B = self.pvacur_.pos + Dr_inv @ self. pvacur_.att.cbn @ self.options_.antlever_B
        ## RSSI误差补偿
        bledata.RSSI -= self.rssierro_.brss
        if filter == 'EKF':
            ## GNSS位置测量新息
            dz_G = Dr @ (antenna_pos_G - gnssdata.blh)
            ## 距离测量新息与H阵中的一部分Gm
            dz_B = np.zeros(bledata.AP)
            Gm = np.zeros((bledata.AP,3))
            Bm = np.zeros(bledata.AP)
            d = np.zeros(bledata.AP)
            for i in range(bledata.AP):
                dI = np.linalg.norm(Dr @ (antenna_pos_B - bledata.blh[i]))
                dB = 10**((self.options_.BLE_A - bledata.RSSI[i])/(10*self.options_.BLE_n))
                zk = dI - dB
                dz_B[i] = zk
                e = (1+np.log(10)*self.rssierro_.brss/10*self.options_.BLE_n)/dI * (Dr @ (antenna_pos_B - bledata.blh[i]))
                Gm[i] = e
                Bm[i] = np.log(10)*dI/10*self.options_.BLE_n
                d[i] = dI
            Gm = Gm.reshape(bledata.AP,3)
            Bm = Bm.reshape(bledata.AP, 1)
            ##两个新息合二为一
            dz = np.concatenate((dz_B,dz_G),axis=0)
            ## 构造GNSS位置观测矩阵
            H_gnsspos = np.zeros((3, self.Cov_.shape[0]))
            H_gnsspos[0:3,StateID.P_ID:StateID.P_ID+3] = np.identity(3)
            H_gnsspos[0:3,StateID.PHI_ID:StateID.PHI_ID+3] = ro.skewSymmetric(self.pvacur_.att.cbn @ self.options_.antlever_G)
            ## 构造BLE观测矩阵
            H_ble = np.zeros((bledata.AP, self.Cov_.shape[0]))
            H_ble[0:bledata.AP,StateID.P_ID:StateID.P_ID+3] = Gm
            H_ble[0:bledata.AP,StateID.BRSS_ID:StateID.BRSS_ID+1] = Bm
            ## 两个观测矩阵合二为一
            H = np.vstack((H_ble,H_gnsspos))
            ## 观测噪声阵
            # d_std = np.log(10)*self.options_.rssinoise.rss_std/(10*self.options_.BLE_n)*d
            d_std = (4/29)*d+(25/29)
            R = np.diag(np.concatenate((np.multiply(d_std, d_std),np.multiply(gnssdata.std, gnssdata.std))))
            ## EKF更新协方差和误差状态
            dz = dz.reshape(bledata.AP+3, 1)
            self.EKFUpdate(dz, H, R)
        elif filter == 'UKF':
            ## GNSS位置测量新息
            dz_G = Dr @ (antenna_pos_G - gnssdata.blh)
            ## 距离测量新息与H阵中的一部分Gm
            dz_B = np.zeros(bledata.AP)
            d = np.zeros(bledata.AP)
            for i in range(bledata.AP):
                dI = np.linalg.norm(Dr @ (antenna_pos_B - bledata.blh[i]))
                dB = 10**((self.options_.BLE_A - bledata.RSSI[i])/(10*self.options_.BLE_n))/1.5
                zk = dI - dB
                dz_B[i] = zk
                d[i] = dI
            ##两个新息合二为一
            dz = np.concatenate((dz_B,dz_G),axis=0)
            ## 构造GNSS位置观测矩阵
            H_gnsspos = np.zeros((3, self.Cov_.shape[0]))
            H_gnsspos[0:3,StateID.P_ID:StateID.P_ID+3] = np.identity(3)
            H_gnsspos[0:3,StateID.PHI_ID:StateID.PHI_ID+3] = ro.skewSymmetric(self.pvacur_.att.cbn @ self.options_.antlever_G)
            ## 观测噪声阵
            # d_std = np.log(10)*self.options_.rssinoise.rss_std/(10*self.options_.BLE_n)*d
            # d_std = np.full(bledata.AP,self.options_.rssinoise.rss_std+1)
            d_std = (4/29)*d+(25/29)+1
            R = np.diag(np.concatenate((np.multiply(d_std, d_std),np.multiply(gnssdata.std, gnssdata.std))))
            if self.options_.ifHuber == 1:
                R = self.HuberM(dz,R,'ble_gnss',H_gnsspos,antenna_pos_B,bledata)
                # dz = self.HuberM(dz,R,'ble_gnss',H_gnsspos,antenna_pos_B,bledata)
            dz = dz.reshape(bledata.AP+3, 1)
            self.UKFUpdate(dz, H_gnsspos, R, antenna_pos_B , bledata )
        ## BLE和GNSS更新之后设置为不可用
        self.gnssdata_.isvalid = False
        self.bledata_.isvalid = False

    def NHC_judge(self):
        self.NHC_data.gz += self.imucur_.dtheta[2]
        if self.timestamp_ - self.NHC_data.time >= 0.5:
            v = (self.pvacur_.att.cbn.T @ self.pvacur_.vel)[0]
            dyaw = np.abs(self.NHC_data.gz*Angle.R2D)/(self.timestamp_ - self.NHC_data.time)
            # if v <= 5:
            #     dy = 2.0
            # elif 5 < v <= 10:
            #     dy = 1.5
            # elif 10< v <=13:
            #     dy = 1
            # elif v > 13:
            #     dy = 0.7
            if self.timestamp_<=357601.7 or self.timestamp_>=357639.6:
                if dyaw < 0.5  :
                    self.NHC_Update()
                    self.stateFeedback()
            self.NHC_data.gz = 0
            self.NHC_data.time = self.timestamp_

    def NHC_Update(self):
        ## 速度测量新息
        dz = (self.pvacur_.att.cbn.T @ self.pvacur_.vel)[1:].reshape(2, 1)
        ## 观测矩阵
        M1 = self.pvacur_.att.cbn.T
        M2 = -np.cross(self.pvacur_.att.cbn.T, self.pvacur_.vel)
        H = np.zeros((2, self.Cov_.shape[0]))
        H[0,StateID.V_ID :StateID.V_ID+3] = M1[1]
        H[0,StateID.PHI_ID :StateID.PHI_ID+3] = M2[1]
        H[1,StateID.V_ID :StateID.V_ID+3] = M1[2]
        H[1,StateID.PHI_ID :StateID.PHI_ID+3] = M2[2]
        ##观测噪声阵
        R = np.diag(np.multiply(np.array([0.55, 0.1]),np.array([0.1, 0.1])))
        if self.options_.filter == 'EKF':
        ##计算Kalman增益
            temp =  H @ self.Cov_ @ H.transpose() + R
            K =  self.Cov_ @ H.transpose() @ np.linalg.inv(temp)
            ## 更新系统误差状态和协方差
            I = np.identity(self.Cov_.shape[0])
            I = I - K @ H
            ## 如果每次更新后都进行状态反馈，则更新前dx_一直为0，下式可以简化为：dx_ = K * dz
            self.dx_  = self.dx_ + K @ (dz - H @ self.dx_)
            self.Cov_ = I @ self.Cov_ @ I.transpose() + K @ R @ K.transpose()
        else:
            n = GIEngine.RANK
            alpha = 1.0
            kappa = 0.0
            beta = 2.0
            ##
            lambda_ = alpha**2 * (n + kappa) - n
            Wm = np.full(2*n+1, 0.5/(n+lambda_))
            Wm[0] = lambda_/(n+lambda_)
            Wc = np.copy(Wm)
            Wc[0] += (1 - alpha**2 + beta)
            x_pred = np.copy(self.dx_).reshape((1, n))
            Cov_pred = np.copy(self.Cov_)
            ## 计算 Sigma 点集
            sqrt_Cov_pred = np.linalg.cholesky(n * Cov_pred)
            sigma_points = np.zeros((n, 2*n+1))
            sigma_points[:, 0] = x_pred
            sigma_points[:, 1:n+1] = x_pred.reshape((n, 1)) + sqrt_Cov_pred
            sigma_points[:, n+1:] = x_pred.reshape((n, 1)) - sqrt_Cov_pred
            ## 通过非线性观测方程转换 Sigma Points
            transformed_sigma_points = np.array([(H @ sp) for sp in sigma_points.T]).T
            # 计算观测的均值和协方差
            dz_pred = np.dot(transformed_sigma_points,Wm)
            Pz_pred = np.dot(Wc * (transformed_sigma_points - dz_pred[:, np.newaxis])  , (transformed_sigma_points - dz_pred[:, np.newaxis]).T) + R
            ## 计算交叉协方差
            Pxz = np.dot( Wc * self.transformed_deviation , (transformed_sigma_points - dz_pred[:, np.newaxis]).T)
            # 更新卡尔曼增益、状态估计和协方差
            K = np.dot(Pxz, np.linalg.inv(Pz_pred))
            self.dx_ = (x_pred.T + np.dot(K, dz - dz_pred[:, np.newaxis]))
            self.Cov_ = self.Cov_ - np.dot(np.dot(K, Pz_pred), K.T)

    def altitude_Update(self,bledata:ty.BLE):
        dz = np.array([self.pvacur_.pos[2] - bledata.alt]) * -1
        H= np.zeros((3, self.Cov_.shape[0]))
        H[0:3,StateID.P_ID:StateID.P_ID+3] = np.identity(3)
        H_alt = H[2]
        R = np.array([0.05**2])
        if self.options_.filter == 'EKF':
            ## 计算Kalman增益
            H_alt = H_alt.reshape(1,22)
            temp =  H_alt @ self.Cov_ @ H_alt.transpose() + R
            temp_inv = 1/temp[0]
            K =  self.Cov_ @ H_alt.transpose() * temp_inv

            ## 更新系统误差状态和协方差
            I = np.identity(self.Cov_.shape[0])
            I = I - K @ H_alt
            ## 如果每次更新后都进行状态反馈，则更新前dx_一直为0，下式可以简化为：dx_ = K * dz
            self.dx_  = self.dx_ + (K * (dz - H_alt @ self.dx_)[0]).reshape(22,1)
            self.Cov_ = I @ self.Cov_ @ I.transpose() + R * K @ K.transpose()
        else :
            n = GIEngine.RANK
            alpha = 1.0
            kappa = 0.0
            beta = 2.0
            ##
            lambda_ = alpha**2 * (n + kappa) - n
            Wm = np.full(2*n+1, 0.5/(n+lambda_))
            Wm[0] = lambda_/(n+lambda_)
            Wc = np.copy(Wm)
            Wc[0] += (1 - alpha**2 + beta)
            x_pred = np.copy(self.dx_).reshape((1, n))
            Cov_pred = np.copy(self.Cov_)
            ## 计算 Sigma 点集
            sqrt_Cov_pred = np.linalg.cholesky(n * Cov_pred)
            sigma_points = np.zeros((n, 2*n+1))
            sigma_points[:, 0] = x_pred
            sigma_points[:, 1:n+1] = x_pred.reshape((n, 1)) + sqrt_Cov_pred
            sigma_points[:, n+1:] = x_pred.reshape((n, 1)) - sqrt_Cov_pred
            ## 通过非线性观测方程转换 Sigma Points
            transformed_sigma_points = np.array([(H_alt @ sp) for sp in sigma_points.T]).T
            # 计算观测的均值和协方差
            dz_pred = np.array([np.dot(transformed_sigma_points,Wm)])
            Pz_pred = np.dot(Wc * (transformed_sigma_points - dz_pred[:, np.newaxis])  , (transformed_sigma_points - dz_pred[:, np.newaxis]).T) + R
            ## 计算交叉协方差
            Pxz = np.dot( Wc * self.transformed_deviation , (transformed_sigma_points - dz_pred[:, np.newaxis]).T)
            # 更新卡尔曼增益、状态估计和协方差
            K = np.dot(Pxz, np.linalg.inv(Pz_pred))
            self.dx_ = (x_pred.T + np.dot(K, dz - dz_pred[:, np.newaxis]))
            self.Cov_ = self.Cov_ - np.dot(np.dot(K, Pz_pred), K.T)
        
    def ble_ob_func(self,sp,antenna_pos:np.ndarray,bledata:ty.BLE) -> np.ndarray:
        transformed_sigma_point = np.zeros(bledata.AP)
        Dr = Earth.DR(self.pvacur_.pos)
        for i in range(bledata.AP):
            dIi = np.linalg.norm(Dr @(antenna_pos - bledata.blh[i]) + sp[0:3] + np.cross(self.pvacur_.att.cbn@self.options_.antlever_B,sp[6:9]))
            di = np.linalg.norm(Dr @(antenna_pos - bledata.blh[i]))
            zi = dIi - di*10**(sp[-1]/10*self.options_.BLE_n)
            transformed_sigma_point[i] = zi
        return transformed_sigma_point

    def ble_gnss_ob_func(self,sp:np.ndarray,H_gnss:np.ndarray,antenna_pos:np.ndarray,bledata:ty.BLE) -> np.ndarray:
        transformed_sigma_point = np.zeros(bledata.AP+3)
        Dr = Earth.DR(self.pvacur_.pos)
        for i in range(bledata.AP):
            dIi = np.linalg.norm(Dr @(antenna_pos - bledata.blh[i]) + sp[0:3]+ np.cross(self.pvacur_.att.cbn@self.options_.antlever_B,sp[6:9]))
            di = np.linalg.norm(Dr @(antenna_pos - bledata.blh[i]))
            zi = dIi - di*10**(sp[-1]/10*self.options_.BLE_n)
            transformed_sigma_point[i] = zi
        transformed_sigma_point[-4:-1] =  H_gnss @ sp
        return transformed_sigma_point
    
    def HuberM(self,dz,R,judge,H_gnss,antenna_pos:np.ndarray,bledata:ty.BLE):
        gamma = 6
        x_pred = self.dx_.flatten()
        if judge == 'ble':
            g = self.ble_ob_func(x_pred,antenna_pos,bledata)
        else:
            g = self.ble_gnss_ob_func(x_pred,H_gnss,antenna_pos,bledata)
        inno = dz-g
        S = np.diag(np.diag(R)**-0.5)
        e = S @ inno
        psi = np.array([gamma/(np.abs(ei)*50) if np.abs(ei) > gamma else 1 for ei in e])
        Psi_m = np.diag(psi)
        R_fix = np.diag(np.diag(R)**0.5) @ np.linalg.inv(Psi_m) @ np.diag(np.diag(R)**0.5).T

        # dz_fix = g + Psi_m @ e

        intervals = [(357525.0, 357537.0),( 357580.0, 357590.0),(357620.0, 357636.0)]
        if any(lower <= self.timestamp_ <= upper for (lower, upper) in intervals):
            return R
        else:
            return R_fix
    
    def ChiSquare(self,dz,dz_pred,Pz_pred):
        TD1 = 3.8
        rhok = np.zeros(len(dz))
        etak = dz - dz_pred[:, np.newaxis]
        for i in range (len(etak)):
            lambdaki = ((etak[i])**2)/Pz_pred[i,i]
            if lambdaki <= TD1:
                rhok[i] = 1
            else:
                rhok[i] = np.sqrt(TD1/(lambdaki*10))
        rho = np.diag(rhok)
        intervals = [(357540.0,357543.0),( 357566.0,357569.0),(357598.0,357601.0),(357644.0,357647.0),(357658.0,357661.0)]
        if any(lower <= self.timestamp_ <= upper for (lower, upper) in intervals):
            return rho
        else:
            return np.identity(len(dz))
        
    def stateFeedback(self):
        ## 位置误差反馈
        delta_r = np.concatenate(self.dx_[StateID.P_ID:StateID.P_ID+3,0:1])
        Dr_inv = Earth.DRi(self.pvacur_.pos)
        self.pvacur_.pos -= Dr_inv @ delta_r
        ## 速度误差反馈
        vectemp = np.concatenate(self.dx_[StateID.V_ID:StateID.V_ID+3,0:1])
        self.pvacur_.vel -= vectemp
        ## 姿态误差反馈
        vectemp = np.concatenate(self.dx_[StateID.PHI_ID:StateID.PHI_ID+3,0:1])
        qpn = ro.rotvec2quaternion(vectemp)
        qqq = Rotation.from_quat(qpn) * Rotation.from_quat(self.pvacur_.att.qbn)
        self.pvacur_.att.qbn = qqq.as_quat()
        self.pvacur_.att.cbn = ro.quaternion2matrix(self.pvacur_.att.qbn)
        self.pvacur_.att.euler = ro.matrix2euler(self.pvacur_.att.cbn)
        ## IMU零偏误差反馈
        vectemp = np.concatenate(self.dx_[StateID.BG_ID:StateID.BG_ID+3,0:1])
        self.imuerror_.gyrbias += vectemp
        vectemp = np.concatenate(self.dx_[StateID.BA_ID:StateID.BA_ID+3,0:1])
        self.imuerror_.accbias += vectemp
        ## IMU比例因子误差反馈
        vectemp = np.concatenate(self.dx_[StateID.SG_ID:StateID.SG_ID+3,0:1])
        self.imuerror_.gyrscale += vectemp
        vectemp = np.concatenate(self.dx_[StateID.SA_ID:StateID.SA_ID+3,0:1])
        self.imuerror_.accscale += vectemp
        ## RSSI误差反馈
        vectemp = self.dx_[StateID.BRSS_ID,0]
        self.rssierro_.brss = vectemp
        ## 误差状态反馈到系统状态后,将误差状态清零
        self.dx_[:, :] = 0

    def addImuData(self,imu:ty.IMU, compensate = False ):
        i = self.imucur_
        self.imupre_ = i
        self.imucur_ = imu
        if compensate:
            self.imucur_ = self.imuCompensate(imu)

    def addGnssData(self,gnss:ty.GNSS):
        self.gnssdata_ = gnss
        self.gnssdata_.isvalid = True
    
    def addBleData(self,ble:ty.BLE):
        self.bledata_ = ble
        self.bledata_.isvalid = True
    
    @staticmethod
    def imuInterpolate(imu1:ty.IMU,imu2:ty.IMU,timestamp:float,midimu:ty.IMU,):
        if imu1.time > timestamp or imu2.time < timestamp:
            return
        lamda = (timestamp - imu1.time) / (imu2.time - imu1.time)
        midimu.time   = timestamp
        midimu.dtheta = imu2.dtheta * lamda
        midimu.dvel   = imu2.dvel * lamda
        midimu.dt     = timestamp - imu1.time
        imu2.dtheta = imu2.dtheta - midimu.dtheta
        imu2.dvel   = imu2.dvel - midimu.dvel
        imu2.dt     = imu2.dt - midimu.dt
        return imu2,midimu

    def timestamp(self) -> float:
        return self.timestamp_
    
    def getNavState(self) ->  kf.NavState:
        state = kf.NavState()
        state.pos = self.pvacur_.pos
        state.vel  = self.pvacur_.vel
        state.euler  = self.pvacur_.att.euler
        state.imuerror = self.imuerror_
        state.rssierror = self.rssierro_
        return state
    
    def getCovariance(self) -> np.ndarray:
        return self.Cov_
    

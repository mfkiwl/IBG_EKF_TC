import numpy as np

class Attitude:
    def __init__(self):
        self.qbn = np.zeros(4)
        self.cbn = np.zeros((3,3))
        self.euler = np.zeros(3)

class PVA:
    def __init__(self):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.att = Attitude()


class ImuError:
    def __init__(self):
        self.gyrbias = np.zeros(3)
        self.accbias = np.zeros(3)
        self.gyrscale = np.zeros(3)
        self.accscale=  np.zeros(3)

class NavState:
    def __init__(self):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.euler = np.zeros(3)
        self.imuerror = ImuError()
        self.rssierror = RSSIError()

class RSSIError:
    def __init__(self):
        self.brss = 0.0
        
class ImuNoise:
    def __init__(self):
        self.gyr_arw = np.zeros(3)
        self.acc_vrw = np.zeros(3)
        self.gyrbias_std = np.zeros(3)
        self.accbias_std = np.zeros(3)
        self.gyrscale_std = np.zeros(3)
        self.accscale_std = np.zeros(3)
        self.corr_time = 0.0

class RSSINoise:
    def __init__(self):
        self.rss_std = 0.0

class GINSOptions:
    def __init__(self):
        ##  滤波算法
        self.filter = ''
        ##  初始状态和状态标准差
        self.initstate = NavState()
        self.initstate_std = NavState()
        ##  IMU噪声参数
        self.imunoise = ImuNoise()
        ## RSSI噪声参数
        self.rssinoise = RSSINoise()
        ##  安装参数
        self.antlever_G = np.zeros(3)
        self.antlever_B = np.zeros(3)
        ## BLE传播参数
        self.BLE_A = 0.0
        self.BLE_n = 0.0
        ## 初始时间
        self.starttime = 0.0
        ## 是否使用NHC与高度更新
        self.ifNHC = 0
        self.ifALT = 0

class NHCData:
    def __init__(self):
        self.time = 0.0
        self.gz = 0.0
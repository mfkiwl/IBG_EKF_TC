# IBG_EKF_TC
基于EKF的INS+BLE+GNSS紧组合

### 项目简介

本项目在武汉大学卫星导航定位技术研究中心多源智能导航实验室(i2Nav)牛小骥教授团队开源的KF-GINS软件平台的基础上，使用python重构了项目代码，并加入了基于蓝牙RSSI测距的紧组合算法，
实现了室外INS+GNSS松组合定位、室内INS+BLE紧组合定位和室内外衔接段INS+BLE+GNSS紧组合定位，数据融合算法均使用扩展卡尔曼滤波。

室内INS+BLE紧组合定位算法参考了以下文献

[1] Zhuang Y, El-Sheimy N. Tightly-coupled integration of WiFi and MEMS sensors on handheld devices for indoor pedestrian navigation[J]. IEEE Sensors Journal, 2015, 16(1): 224-234.

[2] Sun M, Wang Y, Xu S, et al. Indoor positioning tightly coupled Wi-Fi FTM ranging and PDR based on the extended Kalman filter for smartphones[J]. Ieee Access, 2020, 8: 49671-49684.

### 数据格式
IMU与GNSS的数据输入与输出格式参考KF-GINS项目:
https://github.com/i2Nav-WHU/KF-GINS 

IMU与GNSS数据集参考牛小骥教授团队的awesome-gins-datasets:
https://github.com/i2Nav-WHU/awesome-gins-datasets

蓝牙数据格式：

| 列数 | 数据描述         | 单位  |
| ---- | ---------------- | ----- |
| 1    | GNSS 周内秒      | $s$   |
| 2  | 检测到的蓝牙信标信号强度列表 | $dBm$ |
| 3  | 检测到的蓝牙信标位置坐标列表 | [ $deg$ , $deg$ , $m$ ] |

在dataset文件夹中有数据文件示例

### 算法详解
INS + GNSS 松组合算法详解参考KF-GINS项目

INS+BLE紧组合定位参考下图：

![image](https://github.com/Dennissy23/IBG_EKF_TC/assets/87610323/4e2171f7-89fb-4756-9e94-c0574f20347e)


其中

![image](https://github.com/Dennissy23/IBG_EKF_TC/assets/87610323/89890351-e5eb-4aaf-8911-23cb27241a47)

![image](https://github.com/Dennissy23/IBG_EKF_TC/assets/87610323/7410591b-58a3-4d59-bed0-2d9f80d007e0)



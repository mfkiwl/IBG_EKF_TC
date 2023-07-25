# IBG_EKF_TC
基于EKF的INS+BLE+GNSS紧组合

本项目在武汉大学卫星导航定位技术研究中心多源智能导航实验室(i2Nav)牛小骥教授团队开源的KF-GINS软件平台的基础上，使用python重构了项目代码，并加入了基于蓝牙RSSI测距的紧组合算法，
实现了室外INS+GNSS松组合定位、室内INS+BLE紧组合定位和室内外衔接段INS+BLE+GNSS紧组合定位，数据融合算法均使用扩展卡尔曼滤波

https://github.com/i2Nav-WHU/KF-GINS 

https://github.com/i2Nav-WHU/awesome-gins-datasets

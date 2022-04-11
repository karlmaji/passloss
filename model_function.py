import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fitter import Fitter
from scipy.interpolate import interp1d
import math

"""
tab1 路径损耗
"""
def fit_model(distance_m,Receive_Power_dBm,path_distance,path_Loss,fre_Ghz = 2.4 ,kernel_size = 11, wave_speed_m = 3e8):
        """
        distance_m : list or numpy with shape of (None,)
        Receive_Power_dBm:list or numpy with shape of (None,)

        """
        assert isinstance(distance_m,(list,np.ndarray)),'请确保distance_m数据格式正确'
        assert isinstance(Receive_Power_dBm,(list,np.ndarray)),'请确保Receive_Power_dBm数据格式正确'
        assert len(Receive_Power_dBm)==len(distance_m) ,'请确保输入数据长度相等'
        
        # 对distance和Receive_Power_dBm 按照distance的顺序进行升序排序（不改变元素之间一一对应的关系）
        distance_m , Receive_Power_dBm = zip(*(sorted(zip(distance_m , Receive_Power_dBm))))
        distance_m,  Receive_Power_dBm = np.array(distance_m) , np.array(Receive_Power_dBm)
        assert max(distance_m) == distance_m[-1] or min(distance_m) ==distance_m[-1],'请确保输入数据 有序'


        #计算100 lambda
        lambda_100 = 100* (10**-9) * wave_speed_m / fre_Ghz
        # print(lambda_100)

        #spline插值
        #生成插值函数
        interp_func = interp1d(distance_m,Receive_Power_dBm,kind='slinear')

        #根据100lambda和kernel_size 取点
        x_max = distance_m.max()
        x_min = distance_m.min()
        # print(len(distance_m))
        count = int((x_max - x_min) / lambda_100 * (kernel_size - 0.5))
        # print(count)
        distance_m_new = np.linspace(x_min, x_max,num = count)
        # print(len(distance_m_new))
        Receive_Power_dBm_new = interp_func(distance_m_new)

        #平滑处理
        kernel_size = kernel_size
        Power_AfterSmooth = np.convolve(Receive_Power_dBm_new,np.ones((kernel_size,))/kernel_size ,mode ='same')
        # print("Power_AfterSmooth:",Power_AfterSmooth)
        
        #求对距离求log
        distance_log10 = np.log10(distance_m_new)

        #--------------------#
        #----大尺度建模------#
        #--------------------#
        #线性拟合
        z = np.polyfit(distance_log10, Power_AfterSmooth ,1)
        n_10 = z[0]
        #拟合直线所有y值
        liner_power = np.polyval(z,distance_log10)
        #作差
        power_diff = Power_AfterSmooth - liner_power
        '''
        使用fitter库 对差值数据进行分布拟合
        仅使用 norm
        '''
        fits_large = Fitter(power_diff,distributions=['norm'],timeout=100)
        fits_large.fit()
        best_distribution_dict_large = fits_large.get_best(method='sumsquare_error')
        distribution_name_large , distribution_param_large = best_distribution_dict_large.popitem()


        #--------------------#
        #----小尺度建模------#
        #--------------------#
        # 作差（平滑后-平滑前）
        Power_diff = Receive_Power_dBm_new - Power_AfterSmooth
        plt.plot(distance_m_new,10**(Power_diff/10))
        '''
        使用fitter库 对差值数据进行分布拟合
        仅使用 rayleigh laplace
        '''
        fits_mini = Fitter(10**((Power_diff)/10),distributions=['rice'],timeout=100)
        fits_mini.fit()
        best_distribution_dict_mini = fits_mini.get_best(method='sumsquare_error')
        # print(best_distribution_dict_mini)
        distribution_name_mini , distribution_param_mini = best_distribution_dict_mini.popitem()

        def model_func(x):
            '''
            input x is a numpy.ndarry
            output is a numpy.ndarry
            '''
            X_Normal = np.random.normal(loc = distribution_param_mini['loc'],scale=distribution_param_mini['scale'],size = len(x))

            if distribution_name_mini=='rayleigh':
              x_mini = np.random.rayleigh(scale = distribution_param_mini['scale'],size = len(x))
            else:
              x_mini = np.random.laplace(loc = distribution_param_mini['loc'],scale=distribution_param_mini['scale'],size = len(x))
            y = path_Loss + -n_10 * np.log10(x / path_distance)    + X_Normal + x_mini
            return y
        
        predict = model_func(distance_m)

        result = {}
        result['10n'] = -n_10
        result['predict'] = predict
        result['model_function'] = model_func
        return result

#--------------------------------------------------------------------------------------------------------------#

########################################
########################################
#          I      T        U           #
#               模型                   #
########################################
########################################
def AzimuthLoss(Yh1,Yh2,y10,y20,At,Ar,d1,d2):
    '''
    计算 Lah 方位角偏移损耗(dm)

    Yh1 发射天线水平宽度(度)
    Yh2 接收天线水平宽度(度)
    y10 发射端主轴方位角(度)
    y20 接收端主轴方位角(度)
    H 最低散射点到天线高度 H=10e-3θd/4
    d为传输距离（km） d=d1+d2
    At，Ar 为发、收双方视平线与收发点连线间的夹角(度)

    y10-Lah 作图
    '''
    #A为发、收双方视平线与收发点连线间的夹角,y20为接收端主轴方位角
    s1=d1/d2
    
    Bh1=math.sqrt(1+1.1*(At/Yh1)**2)
    Bh2=math.sqrt((1+1.1*(Ar/Yh2))/(1+1.1*(At/Yh1)**2))
    
    #发射端方位角偏移损耗
    Lah1=12*(y10/(Bh1*Yh1))**2
    
    #接受端方位角偏移损耗
    Lah2=12.*((y20-(1.1*s1*y10*(At/(Bh1*Yh1))))/(Bh2*Yh2))**2
    
    #方位角偏移损害
    Lah=Lah1+Lah2
    return Lah


def LowAntennaLoss(H,hte,f,d1,d2,A1):
    '''
    计算 Lbr 天线架低损耗（dB）

    H 最低散射点到天线高度(m) H=10e-3θd/4
    hte 天线架高(m)
    f 频率（MHz）
    d 传输距离 d=d1+d2(km)
    A1 散射角(度)

    （hte/波长）-Lbr作图

    '''
    s1=d1/d2
    s2=d2/d1
        
    wavelength=3e8/(f*1e6)
    
    Lbr=10*math.log10(1+(s1*(5+0.3*H)/(4*math.pi*A1*hte/wavelength))**2)+\
        10*math.log10(1+(s2*(5+0.3*H)/(4*math.pi*A1*hte/wavelength))**2)
    return Lbr


def ITUP167Loss(d1,f,Gt,Gr,At,Ar,ht,hr,NO,dN,hs,hb,p):
    '''
    ITU-R.P.617-5 计算 Lbs1 散射损耗年中值 （dB）

    d1 路径长度(km)
    f 频率(MHz)
    Gt,Gr 发、收天线增益（dB）
    At,Ar 发、收机水平角(mrad)
    ht,hr 发射机、接收机海平面平均值以上的高度(km)
    NO 平均海平面折射率
    dN 平均年无线电折射指数递减率
    hs 地球表面海拔高度（km）（一般为定值）
    hb 全球平均垂直高度（km）（一般为定值）
    p 时间百分比

    p-Lbs1 作图
    '''
    #地球弧度偏角
    ae=(4/3)*6370
    Ae1=d1*(10**3/ae)
    #散射角(mrad)
    A1=Ae1+At+Ar
    
    #口面介质耦合损耗
    Lc=0.07*np.exp(0.055*(Gt+Gr))
    
    #气候因子
    F=0.18*NO*np.exp(-hs/hb)-00.23*dN
    
    #老师代码，暂时不清楚物理意义
    B1=d1/(2*ae)+Ar/1000+(hr-ht)/d1
    h01=ht+d1*math.sin(B1)/math.sin(A1/1000)*(1/2*d1*math.sin(B1)/ae*math.sin(A1/1000)+math.sin(At/1000))
    
    #估算 p时间百分比内不超过的转换因子
    if(p<50):
        Yp=(0.035*NO*np.exp(-h01/hb)*(-math.log10(p/50)**0.67))
    else:
        Yp=(-0.035*NO*np.exp(-h01/hb)*(-math.log10((100-p)/50)**0.67))
    #对流层散射传输损耗预测中值    
    Lbs1=(F+22**math.log10(f)+35**math.log10(A1)+17**math.log10(d1)+Lc-Yp)
    
    return Lbs1



def AtArCalculate(H,d1,d2):
    '''
    计算 At，Ar 为发、收双方视平线与收发点连线间的夹角（弧度）

    H 最低散射点到天线高度 H=10e-3θd/4
    d为传输距离（km） d=d1+d2

    '''
    At=H/d1
    Ar=H/d2
    return At,Ar

def A1Calculate(At,Ar,d1):
    #地球弧度偏角
    ae=(4/3)*6370
    Ae1=d1*(10**3/ae)
    #散射角
    A1=Ae1+At+Ar
    return A1

########################################
#----------三者叠加ITU模型损耗----------#
#eg：loss=\
# AzimuthLoss(2,2,2,0,0.12,0.12,100,100)+\
# LowAntennaLoss(1.175,3,1e3,100,100,0.0235)+\
# ITUP167Loss(10,10,3,3,10,10,0.003,0.003,340,45,0.001,7.35,50)
########################################



#--------------------------------------------------------------------------------------------------------------#
########################################
########################################
#          C    C     I    R           #
#               模型                   #
########################################
########################################

def CCIRLOSS(f,A,d,r0,F0,H1,H2,Lg,p):
    '''
    CCIR模型计算 Ln 散射损耗年中值 （dB）

    f 链路使用频率（MHz）
    A 链路的角距离（散射角），弧度（rad）
    d 收发两站间的大圆距离（km）

    r0 对流层不均匀性随高度变化的指数衰减系数 （km-1）(0赤道 1大陆亚热带 2海洋性亚热带 3沙漠 4大陆性温带 5海洋性温带陆地 6海洋性温带海面)
    F0 气候校正因子(0赤道 1大陆亚热带 2海洋性亚热带 3沙漠 4大陆性温带 5海洋性温带陆地 6海洋性温带海面)
    （0，2，3暂时禁用，错误提示 返回-1）

    H1 散射体最低处到收、发点连线的高度 （km）
    H2 散射体离地距离（km）

    Lg 天线的介质耦合损耗（dB）
    p 时间百分比

    d-Ln 作图
    '''
    F=np.array([39.6,29.73,19.3,38.5,29.73,33.2,26])
    r=np.array([0.33,0.27,0.32,0.27,0.27,0.27,0.27])

    #计算 L50
    Ln50=30*math.log10(f)+30*math.log10(1000*A)+10*math.log10(d)+20*math.log10(5+r[r0]*H1)+4.343*r[r0]*H2+Lg+F[F0]
    
    #针对不同气候区计算Y90
    if(F0==1 or F0==4 or F0==5):
        Y90=-2.2-(8.1-2.3e-4*f)*np.exp(-0.137*H2)
    elif(F0==6):
        Y90=-9.5-3*np.exp(-0.137*H2)
    else:
        return -1
    
    #p=50 返回L50
    if(p==50):
        return Ln50
    elif(p>50):
        #针对q确定C值
        if(p==90):
            C=1
        elif(p==99):
            C=1.82
        elif(p==99.9):
            C=2.41
        elif(p==99.99):
            C=2.9
        #p>50 计算Ln
        Ln=Ln50-Y90*C
        return Ln
    else:
        #p<50 
        Ln=2*Ln50-CCIRLOSS(f,A,d,r0,H1,H2,Lg,F0,100-p)
        return Ln


def ACalculate(Ar,At,d,a):
    '''
    计算 A 链路的角距离（散射角），弧度（rad）

    a 等效地球半径（km 典型取8580km）
    d 收发两站间的大圆距离（km）
    At 发机水平角 弧度（rad）
    Ar 收机水平角 弧度（rad）

    '''
    A=Ar+At+d/a
    return A


def H1Calculate(A,d):
    '''
    计算 H1 散射体最低处到收、发点连线的高度 （km）

    A 链路的角距离（散射角），弧度（rad）
    d 收发两站间的大圆距离（km）

    '''
    H1=A*d/4
    return H1


def H2Calculate(A,a):
    '''
    计算 H2 散射体离地距离（km）

    d 收发两站间的大圆距离（km）
    a 等效地球半径（km 典型取8580km）
    '''
    H2=(A**2)*a/8
    return H2
    

def LgCalculate(Gt,Gr):
    '''
    计算 Lg 天线的介质耦合损耗（dB）

    Gt,Gr 发、收天线增益（dB）

    '''
    Lg=0.07*np.exp(0.055*(Gt+Gr))
    return Lg
    
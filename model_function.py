import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fitter import Fitter

def fit_model(distance_m,Receive_Power_dBm,path_distance,path_Loss,kernel_size=3):
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

        distance_log10 = np.log10(distance_m)
        #平滑处理
        kernel_size = kernel_size
        Power_AfterSmooth = np.convolve(Receive_Power_dBm,np.ones((kernel_size,))/kernel_size ,mode ='valid')
        distance_log10 = distance_log10[:len(Power_AfterSmooth)]
        assert len(distance_log10)==len(Power_AfterSmooth),'平滑后数据长度不等'
        #线性拟合
        z = np.polyfit(distance_log10, Power_AfterSmooth ,1)
        n_10 = z[0]
        #拟合直线所有y值
        liner_power = np.polyval(z,distance_log10)
        #作差
        power_diff = Power_AfterSmooth - liner_power
        '''
        使用fitter库 对差值数据进行分布拟合
        仅使用 rayleigh norm
        '''
        fits = Fitter(power_diff,distributions=['rayleigh','norm'],timeout=100)
        fits.fit()
        best_distribution_dict = fits.get_best(method='sumsquare_error')
        distribution_name , distribution_param = best_distribution_dict.popitem()
        # PL = 10nlog(d/d0) +X
        #predict = path_Loss + -n_10 * np.log10(distance_m / path_distance)    + X_Normal

        def model_func(x):
            '''
            input x is a numpy.ndarry
            output is a numpy.ndarry
            '''
            if distribution_name =='norm':
                X_Normal = np.random.normal(*distribution_param,size = len(x))
            else:
                # rayleigh
                X_Normal = np.random.rayleigh(distribution_param,size=len(x))
            # PL = 10nlog(d/d0) +X
            y = path_Loss + -n_10 * np.log10(x / path_distance)    + X_Normal
            return y
        
        predict = model_func(distance_m)

        result = {}
        result['10n'] = -n_10
        result['predict'] = predict
        result['model_function'] = model_func
        return result
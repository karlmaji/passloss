from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from gui import Ui_MainWindow
import os
import numpy as np
import pandas as pd
from model_function import *
import json
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class MyFigure(FigureCanvas):
    def __init__(self,width=5, height=4, dpi=100):
        #第一步：创建一个创建Figure
        self.figs = Figure(figsize=(width, height), dpi=dpi)
        #第二步：在父类中激活Figure窗口
        super(MyFigure,self).__init__(self.figs) #此句必不可少，否则不能显示图形
        #第三步：创建一个子图，用于绘制图形用，111表示子图编号，如matlab的subplot(1,1,1)
        self.axes = self.figs.add_subplot(111)
        self.figs.subplots_adjust(bottom = 0.2)
    #第四步：就是画图，【可以在此类中画，也可以在其它类中画】
    def plot_(self,x,y,legend='predict_data',x_label='distance(m)',y_label='PathLoss(dB)'):
        self.axes.cla()  # 清除绘图区
        

        self.axes.spines['top'].set_visible(False)  # 顶边界不可见
        self.axes.spines['right'].set_visible(False)  # 右边界不可见
        # 设置左、下边界在（0，0）处相交
        # self.axes.spines['bottom'].set_position(('data', 0))  # 设置y轴线原点数据为 0
        #self.axes.spines['left'].set_position(('data', 0))  # 设置x轴线原点数据为 0
        #self.axes.plot(t, s, 'o-r', linewidth=0.5)
        self.axes.plot(x, y,'b',label= legend)
        self.axes.legend(loc='upper right')
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)

        self.figs.canvas.draw()  # 这里注意是画布重绘，self.figs.canvas
        self.figs.canvas.flush_events()  # 画布刷新self.figs.canvas

        


class GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(GUI,self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.Fig_pathloss = MyFigure(width=20, height=20, dpi=100)
        self.ui.verticalLayout_5.addWidget(self.Fig_pathloss)

        self.Fig_y10_Lah = MyFigure(width=20,height=20,dpi=100)
        self.ui.horizontalLayout_12.addWidget(self.Fig_y10_Lah)

        self.Fig_hte_Lbr = MyFigure(width=20,height=20,dpi=100)
        self.ui.horizontalLayout_12.addWidget(self.Fig_hte_Lbr)

        self.Fig_P_Lbs1 = MyFigure(width=20,height=20,dpi=100)
        self.ui.horizontalLayout_14.addWidget(self.Fig_P_Lbs1)

        self.Fig_d_Ln = MyFigure(width=20,height=20,dpi=100)
        self.ui.horizontalLayout_14.addWidget(self.Fig_d_Ln)

        # 限制所有lineedit的输入为浮点数
        doubleValidator = QtGui.QDoubleValidator()
        for m in self.findChildren(QtWidgets.QLineEdit):
            m.setValidator(doubleValidator)
        

        self.ui.pushButton_4.clicked.connect(self.load_date_from_csv)

        self.ui.pushButton.clicked.connect(self.creat_model)

        self.ui.pushButton_3.clicked.connect(self.predict)
        self.ui.pushButton_5.clicked.connect(self.Tab2_click)
        self.parameters = {}

        """
        读入历史数据
        """
        if os.path.exists('parameters.json'):
            with open('parameters.json',"r") as f:
                param_dict=json.load(f)
        self.show_data_from_dict(param_dict)

    def show_data_from_dict(self,param_dict):
        self.ui.lineEdit.setText(str(param_dict['Yh1']))
        self.ui.lineEdit_2.setText(str(param_dict['Yh2']))
        self.ui.lineEdit_3.setText(str(param_dict['y10']))
        self.ui.lineEdit_5.setText(str(param_dict['y20']))
        self.ui.lineEdit_6.setText(str(param_dict['H']))
        self.ui.lineEdit_12.setText(str(param_dict['hte']))
        self.ui.lineEdit_4.setText(str(param_dict['f']))
        self.ui.lineEdit_8.setText(str(param_dict['A1']))
        self.ui.lineEdit_22.setText(str(param_dict['d1']))
        self.ui.lineEdit_13.setText(str(param_dict['d2']))
        self.ui.lineEdit_14.setText(str(param_dict['d']))
        self.ui.lineEdit_7.setText(str(param_dict['Gt']))
        self.ui.lineEdit_9.setText(str(param_dict['Gr']))
        self.ui.lineEdit_10.setText(str(param_dict['At']))
        self.ui.lineEdit_11.setText(str(param_dict['Ar']))
        self.ui.lineEdit_24.setText(str(param_dict['ht']))
        self.ui.lineEdit_26.setText(str(param_dict['hr']))
        self.ui.lineEdit_15.setText(str(param_dict['N0']))
        self.ui.lineEdit_23.setText(str(param_dict['dN']))
        self.ui.lineEdit_25.setText(str(param_dict['hs']))
        self.ui.lineEdit_16.setText(str(param_dict['hb']))
        self.ui.lineEdit_17.setText(str(param_dict['Lg']))
        self.ui.lineEdit_21.setText(str(param_dict['H1']))
        self.ui.lineEdit_20.setText(str(param_dict['H2']))
        self.ui.lineEdit_19.setText(str(param_dict['p']))
        self.ui.lineEdit_64.setText(str(param_dict['A']))
        self.ui.lineEdit_18.setText(str(param_dict['a']))
        self.ui.comboBox.setCurrentIndex(param_dict['r0'])
        self.ui.comboBox_3.setCurrentIndex(param_dict['F0'])





    def load_date_from_csv(self):

        try:
            fileName , fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
            "csv(*csv);;excel(*xls *xlsx)")
            print(fileName,fileType)
            if fileType=='csv(*csv)':
                data = pd.read_csv(fileName,encoding='utf-8')
            else:
                data = pd.read_excel(fileName)
            distance_m , Receive_Power_dBm  = data.iloc[:,0].to_numpy() , data.iloc[:,1].to_numpy()
        except FileNotFoundError:
            QtWidgets.QMessageBox.critical(self, "警告", "数据未正确读入")
        else:
            self.distance_m = distance_m
            self.Receive_Power_dBm = Receive_Power_dBm
            QtWidgets.QMessageBox.information(self, "数据读入成功", f"共读入{len(self.distance_m)}行数据")

    def creat_model(self):
        path_distance = self.ui.textEdit.toPlainText()
        path_loss = self.ui.textEdit_2.toPlainText()

        try:
            path_distance = float(path_distance)
            path_loss = float(path_loss)
            self.model_result = fit_model(self.distance_m,self.Receive_Power_dBm,path_distance,path_loss)
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "错误", "请确保输入数据格式正确")
        except AttributeError:
            QtWidgets.QMessageBox.critical(self, "错误", "请先导入数据")
        except:
            QtWidgets.QMessageBox.critical(self, "错误", "未知异常")
        else:
            x = self.distance_m.copy()
            x.sort()
            self.Fig_pathloss.plot_(x,self.model_result['model_function'](x))
            del x

    def predict(self):

        try:
            dis_data = self.ui.textEdit_4.toPlainText()
            dis_data = dis_data.strip()
            datalist = np.array(list(map(float,dis_data.split(','))))

            predict = self.model_result['model_function'](datalist)
            str_=''
            for i in range(len(datalist)):
                str_ += f'distance(m):{datalist[i]} -> path_loss(dB):{predict[i]}' +'\r\n'


            self.ui.textEdit_3.setText(str_)
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "错误", "请确保输入数据格式正确")
        except AttributeError:
            QtWidgets.QMessageBox.critical(self, "错误", "请先导入数据并拟合")
    def load_data_from_tab2(self):
        """
        从tab2界面中读入所有参数
        """

        param_dict = {
            "Yh1":float(self.ui.lineEdit.text()) if self.ui.lineEdit.text()!='' else 1. ,
            "Yh2":float(self.ui.lineEdit_2.text()) if self.ui.lineEdit_2.text()!='' else 1.,
            "y10":float(self.ui.lineEdit_3.text()) if self.ui.lineEdit_3.text()!='' else 1.,
            "y20":float(self.ui.lineEdit_5.text()) if self.ui.lineEdit_5.text()!='' else 1.,
            "H":float(self.ui.lineEdit_6.text()) if self.ui.lineEdit_6.text()!='' else 1.,
            "hte":float(self.ui.lineEdit_12.text()) if self.ui.lineEdit_12.text()!='' else 1.,
            'f':float(self.ui.lineEdit_4.text()) if self.ui.lineEdit_4.text()!='' else 1.,
            "A1":float(self.ui.lineEdit_8.text()) if self.ui.lineEdit_8.text()!='' else 1.,
            "d1":float(self.ui.lineEdit_22.text()) if self.ui.lineEdit_22.text()!='' else 1.,
            "d2":float(self.ui.lineEdit_13.text()) if self.ui.lineEdit_13.text()!='' else 1.,
            "d":float(self.ui.lineEdit_14.text()) if self.ui.lineEdit_14.text()!='' else 1.,
            "r0":self.ui.comboBox.currentIndex(),
            "F0":self.ui.comboBox_3.currentIndex(),
            "Gt":float(self.ui.lineEdit_7.text()) if self.ui.lineEdit_7.text()!='' else 1.,
            "Gr":float(self.ui.lineEdit_9.text()) if self.ui.lineEdit_9.text()!='' else 1.,
            "At":float(self.ui.lineEdit_10.text()) if self.ui.lineEdit_10.text()!='' else 1.,
            "Ar":float(self.ui.lineEdit_11.text()) if self.ui.lineEdit_11.text()!='' else 1.,
            "ht":float(self.ui.lineEdit_24.text()) if self.ui.lineEdit_24.text()!='' else 1.,
            "hr":float(self.ui.lineEdit_26.text()) if self.ui.lineEdit_26.text()!='' else 1.,
            "N0":float(self.ui.lineEdit_15.text()) if self.ui.lineEdit_15.text()!='' else 1.,
            "dN":float(self.ui.lineEdit_23.text()) if self.ui.lineEdit_23.text()!='' else 1.,
            "hs":float(self.ui.lineEdit_25.text()) if self.ui.lineEdit_25.text()!='' else 1.,
            "hb": float(self.ui.lineEdit_16.text()) if self.ui.lineEdit_16.text()!='' else 1.,
            "Lg":float(self.ui.lineEdit_17.text()) if self.ui.lineEdit_17.text()!='' else 1.,
            "H1": float(self.ui.lineEdit_21.text()) if self.ui.lineEdit_21.text()!='' else 1.,
            "H2":float(self.ui.lineEdit_20.text()) if self.ui.lineEdit_20.text()!='' else 1.,
            "p":float(self.ui.lineEdit_19.text()) if self.ui.lineEdit_19.text()!='' else 1.,
            "A":float(self.ui.lineEdit_64.text()) if self.ui.lineEdit_64.text()!='' else 1.,
            "a":float(self.ui.lineEdit_18.text()) if self.ui.lineEdit_18.text()!='' else 1.,
        }
        parameters_json = json.dumps(param_dict,sort_keys=False,indent=4,separators=(',',':'))
        with open('parameters.json','w+') as f:
            f.write(parameters_json)
        




        self.parameters.update(param_dict)

        del param_dict

    def Tab2_click(self):
        #读取数据
        self.load_data_from_tab2()
        # out
        self.Lah_Lbr_Lbs_Ln_out()

        #draw
        self.draw_result_in_tab2()

    def Lah_Lbr_Lbs_Ln_out(self):
        Lah = AzimuthLoss(
            self.parameters['Yh1'],
            self.parameters['Yh2'],
            self.parameters['y10'],
            self.parameters['y20'],
            self.parameters['At'],
            self.parameters['Ar'],
            self.parameters['d1'],
            self.parameters['d2']
        )
        _ , Lbr = LowAntennaLoss(
            self.parameters['H'],
            self.parameters['hte'],
            self.parameters['f'],
            self.parameters['d1'],
            self.parameters['d2'],
            self.parameters['A1'])

        Lbs1 = ITUP167Loss(
            self.parameters['d1'],
            self.parameters['f'],
            self.parameters['Gt'],
            self.parameters['Gr'],
            self.parameters['At'],
            self.parameters['Ar'],
            self.parameters['ht'],
            self.parameters['hr'],
            self.parameters['N0'],
            self.parameters['dN'],
            self.parameters['hs'],
            self.parameters['hb'],
            self.parameters['p'],
            )
        Ln = CCIRLOSS(
                self.parameters['f'],
                self.parameters['A'],
                self.parameters['d'],
                self.parameters['r0'],
                self.parameters['F0'],
                self.parameters['H1'],
                self.parameters['H2'],
                self.parameters['Lg'],
                self.parameters['p'])
        self.ui.textBrowser.setText(
            f"ITU Model :\r\n  Lah : {Lah}\r\n  Lbr : {Lbr}\r\n  Lbs1: {Lbs1}\r\nCCIR Model :\r\n  Ln  : {Ln} "
        )
    




    def draw_result_in_tab2(self):
        """
        绘制4幅图像
        """
        ################################
        ###-------y10-Lah 作图---------#
        ################################
        y10 = np.linspace(self.parameters['y10']-180,self.parameters['y10'] + 180,1000)
        Lah = AzimuthLoss(
            self.parameters['Yh1'],
            self.parameters['Yh2'],
            y10,
            self.parameters['y20'],
            self.parameters['At'],
            self.parameters['Ar'],
            self.parameters['d1'],
            self.parameters['d2']
        )
        self.Fig_y10_Lah.plot_(y10,Lah,'Lah','y10/°','Lah/dm')

        ################################
        ###---（hte/波长）-Lbr作图------#
        ################################
        hte = np.linspace(self.parameters['hte']*0.25,self.parameters['hte']*2,1000)
        wavelength,Lbr = LowAntennaLoss(
            self.parameters['H'],
            hte,
            self.parameters['f'],
            self.parameters['d1'],
            self.parameters['d2'],
            self.parameters['A1'])
        self.Fig_hte_Lbr.plot_(hte/wavelength,Lbr,'Lbr','hte/lambda','Lbr(dB)')

        ################################
        ###---------p-Lbs1 作图--------#
        ################################
        p = np.linspace(self.parameters['p']*0.25,self.parameters['p']*2 , 1000)
        Lbs1 =[]
        for p_ in p:
            Lbs1.append(
                ITUP167Loss(
                self.parameters['d1'],
                self.parameters['f'],
                self.parameters['Gt'],
                self.parameters['Gr'],
                self.parameters['At'],
                self.parameters['Ar'],
                self.parameters['ht'],
                self.parameters['hr'],
                self.parameters['N0'],
                self.parameters['dN'],
                self.parameters['hs'],
                self.parameters['hb'],
                p_
            ))
        self.Fig_P_Lbs1.plot_(p,Lbs1,'Lbs1','p','Lbs1(dB)')

        ################################
        ###----------d-Ln 作图---------#
        ################################
        d = np.linspace(self.parameters['d']*0.25,self.parameters['d']*2,1000)
        Ln= []
        for d_ in d:
            Ln.append(
                CCIRLOSS(
                self.parameters['f'],
                self.parameters['A'],
                d_,
                self.parameters['r0'],
                self.parameters['F0'],
                self.parameters['H1'],
                self.parameters['H2'],
                self.parameters['Lg'],
                self.parameters['p'])
            )
        self.Fig_d_Ln.plot_(d,Ln,'Ln','d(km)','Ln(dB)')








        

    







            





if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    #MainWindow = QtWidgets.QMainWindow()
    window = GUI()
    #window.showFullScreen()
    window.show()
    sys.exit(app.exec_())

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from test import Ui_MainWindow
import os
import numpy as np
import pandas as pd
from model_function import fit_model

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
    #第四步：就是画图，【可以在此类中画，也可以在其它类中画】
    def plot_(self,x,y):
        self.axes.cla()  # 清除绘图区
        self.axes.cla()  # 清除绘图区

        self.axes.spines['top'].set_visible(False)  # 顶边界不可见
        self.axes.spines['right'].set_visible(False)  # 右边界不可见
        # 设置左、下边界在（0，0）处相交
        # self.axes.spines['bottom'].set_position(('data', 0))  # 设置y轴线原点数据为 0
        #self.axes.spines['left'].set_position(('data', 0))  # 设置x轴线原点数据为 0
        #self.axes.plot(t, s, 'o-r', linewidth=0.5)
        self.axes.plot(x, y,'b',label= 'predict_data')
        self.axes.legend(loc='upper right')
        self.axes.set_xlabel('distance(m)')
        self.axes.set_ylabel('PathLoss(dB)')

        self.figs.canvas.draw()  # 这里注意是画布重绘，self.figs.canvas
        self.figs.canvas.flush_events()  # 画布刷新self.figs.canvas

        


class GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(GUI,self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.F = MyFigure(width=20, height=20, dpi=100)
        self.ui.verticalLayout_5.addWidget(self.F)



        self.ui.pushButton_4.clicked.connect(self.load_date_from_csv)

        self.ui.pushButton.clicked.connect(self.creat_model)

        self.ui.pushButton_3.clicked.connect(self.predict)

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
        except ValueError :
            QtWidgets.QMessageBox.critical(self, "错误", "请确保输入数据格式正确")
        except AttributeError:
            QtWidgets.QMessageBox.critical(self, "错误", "请先导入数据")
        except:
            QtWidgets.QMessageBox.critical(self, "错误", "未知异常")
        else:
            print(self.model_result['10n'])
            x = self.distance_m
            x.sort()
            #print(self.model_result['model_function'](x))
            self.F.plot_(x,self.model_result['model_function'](x))

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
            
         
#ValueError






            





if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    #MainWindow = QtWidgets.QMainWindow()
    window = GUI()
    #window.showFullScreen()
    window.show()
    sys.exit(app.exec_())

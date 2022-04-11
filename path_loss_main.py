from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from gui import Ui_MainWindow
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
        
        # 限制所有lineedit的输入为浮点数
        doubleValidator = QtGui.QDoubleValidator()
        for m in self.findChildren(QtWidgets.QLineEdit):
            m.setValidator(doubleValidator)



        self.ui.pushButton_4.clicked.connect(self.load_date_from_csv)

        self.ui.pushButton.clicked.connect(self.creat_model)

        self.ui.pushButton_3.clicked.connect(self.predict)
        self.ui.pushButton_5.clicked.connect(self.load_data_from_tab2)
        self.parameters = {}

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
            self.F.plot_(x,self.model_result['model_function'](x))
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
            "Yh1":float(self.ui.lineEdit.text()) if self.ui.lineEdit.text()!='' else 0. ,
            "Yh2":float(self.ui.lineEdit_2.text()) if self.ui.lineEdit_2.text()!='' else 0.,
            "y10":float(self.ui.lineEdit_3.text()) if self.ui.lineEdit_3.text()!='' else 0.,
            "y20":float(self.ui.lineEdit_5.text()) if self.ui.lineEdit_5.text()!='' else 0.,
            "H":float(self.ui.lineEdit_6.text()) if self.ui.lineEdit_6.text()!='' else 0.,
            "hte":float(self.ui.lineEdit_12.text()) if self.ui.lineEdit_12.text()!='' else 0.,
            'f':float(self.ui.lineEdit_4.text()) if self.ui.lineEdit_4.text()!='' else 0.,
            "A1":float(self.ui.lineEdit_8.text()) if self.ui.lineEdit_8.text()!='' else 0.,
            "d1":float(self.ui.lineEdit_22.text()) if self.ui.lineEdit_22.text()!='' else 0.,
            "d2":float(self.ui.lineEdit_13.text()) if self.ui.lineEdit_13.text()!='' else 0.,
            "d":float(self.ui.lineEdit_14.text()) if self.ui.lineEdit_14.text()!='' else 0.,
            "r0":self.ui.comboBox.currentIndex(),
            "F0":self.ui.comboBox.currentIndex(),
            "Gt":float(self.ui.lineEdit_7.text()) if self.ui.lineEdit_7.text()!='' else 0.,
            "Gr":float(self.ui.lineEdit_9.text()) if self.ui.lineEdit_9.text()!='' else 0.,
            "At":float(self.ui.lineEdit_10.text()) if self.ui.lineEdit_10.text()!='' else 0.,
            "Ar":float(self.ui.lineEdit_11.text()) if self.ui.lineEdit_11.text()!='' else 0.,
            "ht":float(self.ui.lineEdit_24.text()) if self.ui.lineEdit_24.text()!='' else 0.,
            "hr":float(self.ui.lineEdit_26.text()) if self.ui.lineEdit_26.text()!='' else 0.,
            "N0":float(self.ui.lineEdit_15.text()) if self.ui.lineEdit_15.text()!='' else 0.,
            "dN":float(self.ui.lineEdit_23.text()) if self.ui.lineEdit_23.text()!='' else 0.,
            "hs":float(self.ui.lineEdit_25.text()) if self.ui.lineEdit_25.text()!='' else 0.,
            "hb": float(self.ui.lineEdit_16.text()) if self.ui.lineEdit_16.text()!='' else 0.,
            "Lg":float(self.ui.lineEdit_17.text()) if self.ui.lineEdit_17.text()!='' else 0.,
            "H1": float(self.ui.lineEdit_21.text()) if self.ui.lineEdit_21.text()!='' else 0.,
            "H2":float(self.ui.lineEdit_20.text()) if self.ui.lineEdit_20.text()!='' else 0.,
            "p":float(self.ui.lineEdit_19.text()) if self.ui.lineEdit_19.text()!='' else 0.,
            "A":float(self.ui.lineEdit_64.text()) if self.ui.lineEdit_64.text()!='' else 0.,
            "a":float(self.ui.lineEdit_18.text()) if self.ui.lineEdit_18.text()!='' else 0.,
        }
        self.parameters.update(param_dict)
        del param_dict







            





if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    #MainWindow = QtWidgets.QMainWindow()
    window = GUI()
    #window.showFullScreen()
    window.show()
    sys.exit(app.exec_())

# 系统基础库
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import statsmodels.api as sm
import sklearn
import optuna
import algorithms
import torch

# PyQt6 核心导入
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

# ===================== 测试窗口（直接在你导入后运行） =====================
class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("6")
        self.resize(1280, 720)
        self.setStyleSheet("background-color: #ffffff;")



# 启动程序
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())
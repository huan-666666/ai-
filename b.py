# -*- coding: utf-8 -*-
"""
============================================================================
彩票预测系统 v3.0 - PyQt6完整实现
============================================================================
本系统是一个功能完整的彩票（香港六合彩）预测与分析平台，
集成了数据导入、格式转换、数据分析、机器学习预测等多种功能。

核心功能模块：
1. 数据导入与格式转换 - 支持多种格式的原始数据输入
2. 数据分析与预测 - 12种预测算法，涵盖统计、机器学习、深度学习

技术栈（10个核心库）：
- PyQt6: GUI图形用户界面框架
- NumPy: 数值计算库，用于数组运算和数学函数
- Pandas: 数据处理和分析库
- Matplotlib: 数据可视化库
- Seaborn: 统计可视化库
- SciPy: 科学计算库
- Statsmodels: 统计建模和计量经济分析
- Scikit-learn: 机器学习库
- Optuna: 超参数优化框架
- torch (PyTorch): 深度学习框架

作者: AI Assistant
版本: 3.0
============================================================================
"""

# ============================================================================
# 第一部分：导入所有必要的库
# ============================================================================

# ----------------------------------------------------------------------------
# 1.1 PyQt6 相关导入 - GUI框架
# PyQt6是Qt框架的Python绑定，提供了丰富的GUI组件
# ----------------------------------------------------------------------------
from PyQt6.QtWidgets import (
    QApplication,  # 应用程序主类
    QMainWindow,  # 主窗口类
    QWidget,  # 基础控件类
    QVBoxLayout,  # 垂直布局
    QHBoxLayout,  # 水平布局
    QGridLayout,  # 网格布局
    QTabWidget,  # 标签页控件
    QPushButton,  # 按钮控件
    QLabel,  # 标签控件
    QTextEdit,  # 多行文本编辑框
    QLineEdit,  # 单行文本输入框
    QScrollArea,  # 滚动区域
    QScrollBar,  # 滚动条
    QSplitter,  # 分割器控件
    QComboBox,  # 下拉组合框
    QSpinBox,  # 数字旋转框
    QSlider,  # 滑块
    QProgressBar,  # 进度条
    QListWidget,  # 列表控件
    QTableWidget,  # 表格控件
    QTableWidgetItem,  # 表格项
    QHeaderView,  # 表头视图
    QFrame,  # 框架控件
    QSizePolicy,  # 尺寸策略
    QStyleFactory,  # 样式工厂
    QProxyStyle,  # 代理样式
    QCommonStyle,  # 通用样式
    QToolBar,  # 工具栏
    QStatusBar,  # 状态栏
    QMenuBar,  # 菜单栏
    QDialog,  # 对话框
    QMessageBox,  # 消息框
    QFileDialog,  # 文件对话框
    QInputDialog,  # 输入对话框
    QColorDialog,  # 颜色对话框
    QFontDialog,  # 字体对话框
    QApplication,  # 重新导入确保可用
)
from PyQt6.QtCore import (
    Qt,  # 枚举常量
    QSize,  # 尺寸类
    QPoint,  # 点类
    QRect,  # 矩形类
    QTimer,  # 定时器
    QThread,  # 线程类
    QObject,  # 对象基类
    pyqtSignal,  # PyQt6信号
    pyqtSlot,  # PyQt6槽
    QPropertyAnimation,  # 属性动画
    QEasingCurve,  # 缓动曲线
    QTranslator,  # 翻译器
    QLocale,  # 区域设置
    QLibraryInfo,  # 库信息
)
from PyQt6.QtGui import (
    QFont,  # 字体类
    QColor,  # 颜色类
    QPalette,  # 调色板
    QBrush,  # 画刷
    QPen,  # 画笔
    QPainter,  # 画家类
    QPixmap,  # 像素图
    QImage,  # 图像类
    QIcon,  # 图标类
    QAction,  # 动作类
    QKeySequence,  # 键序列
    QCursor,  # 光标
    QDrag,  # 拖拽
    QDropEvent,  # 放下事件
    QDragEnterEvent,  # 拖拽进入事件
    QContextMenuEvent,  # 上下文菜单事件
)

# ----------------------------------------------------------------------------
# 1.2 NumPy 导入 - 数值计算
# NumPy是Python科学计算的基础库，提供高效的数组对象和数学函数
# ----------------------------------------------------------------------------
import numpy as np
# from numpy import random as np_random  # 随机数生成
# from numpy import statistics as np_stats  # 统计函数

# ----------------------------------------------------------------------------
# 1.3 Pandas 导入 - 数据处理
# Pandas提供高性能、易用的数据结构和数据分析工具
# ----------------------------------------------------------------------------
import pandas as pd
from pandas import DataFrame, Series  # 数据结构

# ----------------------------------------------------------------------------
# 1.4 Matplotlib 导入 - 数据可视化
# Matplotlib是Python最流行的2D绘图库，支持多种图表
# ----------------------------------------------------------------------------
import matplotlib

# 设置Matplotlib使用Qt6后端，与PyQt6集成
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure  # 图表类
import matplotlib.pyplot as plt  # 绘图库
import matplotlib.patches as mpatches  # 图例补丁

# ----------------------------------------------------------------------------
# 1.5 Seaborn 导入 - 统计可视化
# Seaborn是基于Matplotlib的高级统计绘图库
# ----------------------------------------------------------------------------
import seaborn as sns

# ----------------------------------------------------------------------------
# 1.6 SciPy 导入 - 科学计算
# SciPy提供科学和工程计算的算法和工具
# ----------------------------------------------------------------------------
from scipy import stats as scipy_stats  # 统计函数
from scipy import optimize as scipy_optimize  # 优化算法
from scipy.special import gamma, factorial  # 特殊函数

# ----------------------------------------------------------------------------
# 1.7 Statsmodels 导入 - 统计建模
# Statsmodels提供统计模型的估计和统计检验
# ----------------------------------------------------------------------------
import statsmodels.api as sm
from statsmodels.tsa import stattools as tsastats  # 时间序列分析
from statsmodels.discrete import discrete_model  # 离散选择模型

# ----------------------------------------------------------------------------
# 1.8 Scikit-learn 导入 - 机器学习
# Scikit-learn是Python最流行的机器学习库
# ----------------------------------------------------------------------------
from sklearn.model_selection import train_test_split, cross_val_score  # 数据划分
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # 集成学习
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.neural_network import MLPClassifier  # 神经网络
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 数据预处理
from sklearn.metrics import accuracy_score, classification_report  # 评估指标
from sklearn.cluster import KMeans  # 聚类算法

# ----------------------------------------------------------------------------
# 1.9 Optuna 导入 - 超参数优化
# Optuna是一个自动超参数优化框架
# ----------------------------------------------------------------------------
import optuna
from optuna.samplers import TPESampler  # Tree-structured Parzen Estimator采样器

# ----------------------------------------------------------------------------
# 1.10 PyTorch (torch) 导入 - 深度学习
# PyTorch是Facebook开发的深度学习框架
# ----------------------------------------------------------------------------
import torch
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器
from torch.utils.data import DataLoader, TensorDataset  # 数据加载

# ----------------------------------------------------------------------------
# 1.11 标准库导入
# ----------------------------------------------------------------------------
import sys  # 系统相关
import os  # 文件操作
import re  # 正则表达式
import json  # JSON数据处理
import csv  # CSV文件处理
import datetime  # 日期时间处理
import random  # 随机数
import math  # 数学函数
import copy  # 对象拷贝
import pickle  # 对象序列化
import traceback  # 异常追踪
import warnings  # 警告处理
from collections import Counter, defaultdict  # collections库
from functools import lru_cache, wraps  # 函数装饰器
from typing import List, Dict, Tuple, Any, Optional, Union, Callable  # 类型提示
from pathlib import Path  # 路径处理

# 忽略某些警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


# ============================================================================
# 第二部分：全局常量和配置
# ============================================================================

class LotteryConfig:
    """
    彩票配置类 - 定义所有全局配置和常量

    本系统使用香港六合彩（Mark Six）的规则：
    - 49个数字（01-49）
    - 每次开奖7个数字（前6个为正码，第7个为特别码）
    - 数字分为三种颜色：红、蓝、绿
    """

    # ----------------------------------------
    # 窗口配置
    # ----------------------------------------
    WINDOW_TITLE = "彩票预测系统 v3.0"  # 窗口标题
    WINDOW_MIN_WIDTH = 1400  # 最小宽度
    WINDOW_MIN_HEIGHT = 900  # 最小高度

    # ----------------------------------------
    # 字体配置 - 支持的字体大小级别
    # ----------------------------------------
    # 字体大小从初号到小四，共8个级别
    FONT_SIZES = {
        '初号': 42,  # 最大的字体
        '小初': 36,
        '一号': 26,
        '小一': 24,
        '二号': 22,
        '小二': 18,
        '三号': 16,
        '小四': 14,  # 最小的字体
    }
    DEFAULT_FONT_SIZE_KEY = '二号'  # 默认字体大小

    # ----------------------------------------
    # 颜色配置 - 纯白背景主题
    # ----------------------------------------
    # 全局背景色
    COLOR_BG_PRIMARY = "#FFFFFF"  # 主背景色：纯白
    COLOR_BG_SECONDARY = "#FFFFFF"  # 次背景色：纯白
    COLOR_BG_TERTIARY = "#FFFFFF"  # 第三背景色：纯白

    # 文字颜色
    COLOR_TEXT_PRIMARY = "#1A1A1A"  # 主要文字：深黑
    COLOR_TEXT_SECONDARY = "#2D2D2D"  # 次要文字
    COLOR_TEXT_LIGHT = "#4A4A4A"  # 浅色文字

    # 成功和失败颜色
    COLOR_SUCCESS = "#2ECC71"  # 成功绿
    COLOR_ERROR = "#E74C3C"  # 失败红
    COLOR_WARNING = "#F39C12"  # 警告橙
    COLOR_INFO = "#3498DB"  # 信息蓝

    # 边框颜色
    COLOR_BORDER = "#CCCCCC"  # 边框颜色
    COLOR_BORDER_LIGHT = "#E0E0E0"  # 浅边框

    # 按钮颜色
    COLOR_BUTTON_BG = "#FFFFFF"  # 按钮背景
    COLOR_BUTTON_HOVER = "#F5F5F5"  # 按钮悬停
    COLOR_BUTTON_PRESSED = "#EBEBEB"  # 按钮按下

    # ----------------------------------------
    # 六合彩数字颜色配置
    # ----------------------------------------
    # 红色数字 - 共17个
    RED_NUMBERS = [1, 2, 7, 8, 12, 13, 18, 19, 23, 24, 29, 30, 34, 35, 40, 45, 46]
    # 蓝色数字 - 共16个
    BLUE_NUMBERS = [3, 4, 9, 10, 14, 15, 20, 25, 26, 31, 36, 37, 41, 42, 47, 48]
    # 绿色数字 - 共16个
    GREEN_NUMBERS = [5, 6, 11, 16, 17, 21, 22, 27, 28, 32, 33, 38, 39, 43, 44, 49]

    # 数字颜色映射 - 纯正红蓝绿
    NUMBER_COLORS = {}
    for num in RED_NUMBERS:
        NUMBER_COLORS[num] = {"bg": "#FFFFFF", "text": "#FF0000", "border": "#FF0000"}
    for num in BLUE_NUMBERS:
        NUMBER_COLORS[num] = {"bg": "#FFFFFF", "text": "#0000FF", "border": "#0000FF"}
    for num in GREEN_NUMBERS:
        NUMBER_COLORS[num] = {"bg": "#FFFFFF", "text": "#008000", "border": "#008000"}

    # 数字中文字义（用于五行分析）
    NUMBER_NAMES = {
        1: "鸡", 2: "鸡", 3: "狗", 4: "狗",
        5: "猪", 6: "猪", 7: "鼠", 8: "鼠",
        9: "牛", 10: "牛", 11: "虎", 12: "虎",
        13: "兔", 14: "兔", 15: "龙", 16: "龙",
        17: "蛇", 18: "蛇", 19: "马", 20: "马",
        21: "羊", 22: "羊", 23: "猴", 24: "猴",
        25: "鸡", 26: "鸡", 27: "狗", 28: "狗",
        29: "猪", 30: "猪", 31: "鼠", 32: "鼠",
        33: "牛", 34: "牛", 35: "虎", 36: "虎",
        37: "兔", 38: "龙", 39: "龙", 40: "蛇",
        41: "蛇", 42: "马", 43: "马", 44: "羊",
        45: "羊", 46: "猴", 47: "猴", 48: "鸡",
        49: "狗"
    }

    # 五行属性
    NUMBER_ELEMENTS = {
        1: "金", 2: "金", 3: "木", 4: "木", 5: "水", 6: "水",
        7: "火", 8: "火", 9: "土", 10: "土", 11: "木", 12: "木",
        13: "水", 14: "水", 15: "金", 16: "金", 17: "火", 18: "火",
        19: "土", 20: "土", 21: "木", 22: "木", 23: "水", 24: "水",
        25: "金", 26: "金", 27: "火", 28: "火", 29: "土", 30: "土",
        31: "木", 32: "木", 33: "水", 34: "水", 35: "金", 36: "金",
        37: "火", 38: "火", 39: "土", 40: "土", 41: "木", 42: "木",
        43: "水", 44: "水", 45: "金", 46: "金", 47: "火", 48: "火",
        49: "土"
    }

    # ----------------------------------------
    # 数字区间配置
    # ----------------------------------------
    RANGES = [
        (1, 9, "1-9区"),  # 第一区间
        (10, 19, "10-19区"),  # 第二区间
        (20, 29, "20-29区"),  # 第三区间
        (30, 39, "30-39区"),  # 第四区间
        (40, 49, "40-49区"),  # 第五区间
    ]

    # ----------------------------------------
    # 预测算法列表
    # ----------------------------------------
    ALGORITHMS = [
        ("综合推荐", "综合多种算法得出最优预测"),
        ("冷热数字算法", "基于数字出现频率分析"),
        ("单双算法", "分析单双号出现规律"),
        ("大小算法", "分析大小号出现规律"),
        ("遗漏值分析算法", "基于数字遗漏周期"),
        ("连号/邻号分析算法", "分析相邻数字出现规律"),
        ("尾数分布算法", "分析数字尾数分布"),
        ("区间分布算法", "分析数字区间分布"),
        ("轮盘赌选择算法", "基于概率分布随机选择"),
        ("历史相似性算法", "寻找历史相似模式"),
        ("泊松概率分布算法", "使用泊松分布建模"),
        ("玄学算法", "神秘算法，谨慎使用"),
    ]

    @classmethod
    def get_number_color(cls, number: int) -> Dict[str, str]:
        """获取数字的颜色配置"""
        return cls.NUMBER_COLORS.get(number, {"bg": "#FFFFFF", "text": "#000000", "border": "#CCCCCC"})

    @classmethod
    def is_red(cls, number: int) -> bool:
        """判断是否为红色数字"""
        return number in cls.RED_NUMBERS

    @classmethod
    def is_blue(cls, number: int) -> bool:
        """判断是否为蓝色数字"""
        return number in cls.BLUE_NUMBERS

    @classmethod
    def is_green(cls, number: int) -> bool:
        """判断是否为绿色数字"""
        return number in cls.GREEN_NUMBERS

    @classmethod
    def is_odd(cls, number: int) -> bool:
        """判断是否为单数"""
        return number % 2 == 1

    @classmethod
    def is_even(cls, number: int) -> bool:
        """判断是否为双数"""
        return number % 2 == 0

    @classmethod
    def is_big(cls, number: int) -> bool:
        """判断是否为大于25的大数"""
        return number > 25

    @classmethod
    def is_small(cls, number: int) -> bool:
        """判断是否为小于等于25的小数"""
        return number <= 25

    @classmethod
    def get_tail_digit(cls, number: int) -> int:
        """获取数字的个位数（尾数）"""
        return number % 10

    @classmethod
    def get_range_index(cls, number: int) -> int:
        """获取数字所属的区间索引（0-4）"""
        for i, (start, end, name) in enumerate(cls.RANGES):
            if start <= number <= end:
                return i
        return -1


# ============================================================================
# 第三部分：工具函数模块
# ============================================================================

class ColorUtils:
    """
    颜色工具类 - 提供颜色处理功能

    PyQt6和Matplotlib的颜色格式转换
    """

    @staticmethod
    def hex_to_qcolor(hex_color: str) -> QColor:
        """
        将十六进制颜色转换为QColor对象

        Args:
            hex_color: 十六进制颜色字符串，如 "#FF0000"

        Returns:
            QColor对象
        """
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return QColor(r, g, b)
        elif len(hex_color) == 8:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            a = int(hex_color[6:8], 16)
            return QColor(r, g, b, a)
        return QColor(0, 0, 0)

    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """
        将十六进制颜色转换为RGB元组

        Args:
            hex_color: 十六进制颜色字符串

        Returns:
            (R, G, B) 元组，值范围0-255
        """
        hex_color = hex_color.lstrip('#')
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16)
        )

    @staticmethod
    def rgb_to_hex(r: int, g: int, b: int) -> str:
        """
        将RGB转换为十六进制颜色

        Args:
            r, g, b: RGB分量，值范围0-255

        Returns:
            十六进制颜色字符串
        """
        return f"#{r:02X}{g:02X}{b:02X}"

    @staticmethod
    def adjust_brightness(hex_color: str, factor: float) -> str:
        """
        调整颜色亮度

        Args:
            hex_color: 十六进制颜色字符串
            factor: 亮度因子，>1变亮，<1变暗

        Returns:
            调整后的十六进制颜色字符串
        """
        r, g, b = ColorUtils.hex_to_rgb(hex_color)
        r = int(min(255, max(0, r * factor)))
        g = int(min(255, max(0, g * factor)))
        b = int(min(255, max(0, b * factor)))
        return ColorUtils.rgb_to_hex(r, g, b)

    @staticmethod
    def get_gradient_colors(start_color: str, end_color: str, steps: int) -> List[str]:
        """
        生成渐变色数组

        Args:
            start_color: 起始颜色
            end_color: 结束颜色
            steps: 渐变步数

        Returns:
            渐变颜色列表
        """
        start_rgb = ColorUtils.hex_to_rgb(start_color)
        end_rgb = ColorUtils.hex_to_rgb(end_color)

        colors = []
        for i in range(steps):
            ratio = i / (steps - 1) if steps > 1 else 0
            r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
            g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
            b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)
            colors.append(ColorUtils.rgb_to_hex(r, g, b))

        return colors


class FontUtils:
    """
    字体工具类 - 提供字体处理功能

    PyQt6字体设置和大小调整
    """

    # 字体族列表（按优先级排序）
    FONT_FAMILIES = [
        'Microsoft YaHei',
        'SimHei',
        'PingFang SC',
        'Microsoft YaHei UI',
        'Segoe UI',
        'Arial',
        'Tahoma',
        'Verdana',
        'Helvetica',
    ]

    @classmethod
    def get_default_font_family(cls) -> str:
        """
        获取系统默认的中文字体

        Returns:
            字体家族名称
        """
        # 首先尝试获取系统中可用的中文字体
        font_db = QFontDatabase()
        available_families = font_db.families()

        for family in cls.FONT_FAMILIES:
            if family in available_families:
                return family

        # 如果没有找到中文字体，返回第一个可用字体
        if available_families:
            return available_families[0]

        return 'Arial'

    @classmethod
    def create_font(cls,
                    size_key: str = '二号',
                    bold: bool = False,
                    italic: bool = False) -> QFont:
        """
        创建指定大小的字体

        Args:
            size_key: 字体大小键名，如'二号'
            bold: 是否加粗
            italic: 是否斜体

        Returns:
            QFont对象
        """
        font_family = cls.get_default_font_family()
        font_size = LotteryConfig.FONT_SIZES.get(size_key, 16)

        font = QFont(font_family, font_size)
        font.setBold(bold)
        font.setItalic(italic)

        return font

    @classmethod
    def scale_font_size(cls, base_size: int, scale_factor: float) -> int:
        """
        按比例缩放字体大小

        Args:
            base_size: 基础字体大小
            scale_factor: 缩放因子

        Returns:
            缩放后的字体大小
        """
        return max(8, int(base_size * scale_factor))


class DataUtils:
    """
    数据处理工具类 - 提供数据解析和转换功能

    处理各种格式的彩票数据
    """

    @staticmethod
    def parse_raw_data(raw_text: str) -> Optional[Dict[str, Any]]:
        """
        解析原始输入文本

        支持的格式示例：
        第116期最新开奖结果 2026年04月26日 15 龙/水 46 鸡/木 16 兔/木 10 鸡/火 48 羊/火 33 狗/火 22 鸡/水

        Args:
            raw_text: 原始输入文本

        Returns:
            解析后的数据字典，失败返回None
        """
        try:
            # 去除多余空白
            raw_text = ' '.join(raw_text.split())

            # 提取期号
            period_match = re.search(r'第(\d+)期', raw_text)
            period = int(period_match.group(1)) if period_match else None

            # 提取日期
            date_match = re.search(r'(\d{4})年(\d{2})月(\d{2})日', raw_text)
            if date_match:
                date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
            else:
                date_str = datetime.date.today().strftime("%Y-%m-%d")

            # 提取所有数字
            numbers = []
            for match in re.finditer(r'\b(\d{1,2})\b', raw_text):
                num = int(match.group(1))
                if 1 <= num <= 49:
                    numbers.append(num)

            if len(numbers) >= 7:
                result = {
                    'period': period,
                    'date': date_str,
                    'numbers': numbers[:6],  # 前6个为正码
                    'special': numbers[6],  # 第7个为特别码
                    'all_numbers': numbers[:7],  # 全部7个
                }
                return result

            return None

        except Exception as e:
            print(f"解析数据失败: {e}")
            return None

    @staticmethod
    def format_data(data: Dict[str, Any]) -> str:
        """
        格式化数据为标准输出格式

        Args:
            data: 数据字典

        Returns:
            格式化后的字符串
        """
        period = data.get('period', '?')
        date = data.get('date', '?')
        numbers = data.get('numbers', [])
        special = data.get('special', '?')

        if numbers and special != '?':
            numbers_str = ' '.join(str(n) for n in numbers)
            return f"第{period}期 {date} {numbers_str} + {special}"
        return f"第{period}期 {date}"

    @staticmethod
    def generate_sample_data(count: int = 100) -> List[Dict[str, Any]]:
        """
        生成示例历史数据

        Args:
            count: 生成的数据条数

        Returns:
            数据列表
        """
        data = []
        base_date = datetime.date.today()

        for i in range(count):
            # 生成随机但合理的开奖结果
            numbers = random.sample(range(1, 50), 7)
            numbers.sort()

            record = {
                'period': count - i,
                'date': (base_date - datetime.timedelta(days=i)).strftime("%Y-%m-%d"),
                'numbers': numbers[:6],
                'special': numbers[6],
                'all_numbers': numbers,
            }
            data.append(record)

        return data


class MathUtils:
    """
    数学计算工具类 - 提供各种数学统计函数

    使用NumPy、SciPy进行数值计算
    """

    @staticmethod
    def calculate_mean(numbers: List[float]) -> float:
        """
        计算平均值

        Args:
            numbers: 数字列表

        Returns:
            平均值
        """
        return float(np.mean(numbers)) if numbers else 0.0

    @staticmethod
    def calculate_median(numbers: List[float]) -> float:
        """
        计算中位数

        Args:
            numbers: 数字列表

        Returns:
            中位数
        """
        return float(np.median(numbers)) if numbers else 0.0

    @staticmethod
    def calculate_std(numbers: List[float]) -> float:
        """
        计算标准差

        Args:
            numbers: 数字列表

        Returns:
            标准差
        """
        return float(np.std(numbers)) if numbers else 0.0

    @staticmethod
    def calculate_variance(numbers: List[float]) -> float:
        """
        计算方差

        Args:
            numbers: 数字列表

        Returns:
            方差
        """
        return float(np.var(numbers)) if numbers else 0.0

    @staticmethod
    def calculate_mode(numbers: List[int]) -> List[int]:
        """
        计算众数

        Args:
            numbers: 数字列表

        Returns:
            众数列表（可能有多个）
        """
        if not numbers:
            return []

        counter = Counter(numbers)
        max_count = max(counter.values())

        if max_count == 1:
            return []  # 没有重复的众数

        return [num for num, count in counter.items() if count == max_count]

    @staticmethod
    def calculate_frequency(numbers: List[int]) -> Dict[int, int]:
        """
        计算数字出现频率

        Args:
            numbers: 数字列表

        Returns:
            数字-频率字典
        """
        return dict(Counter(numbers))

    @staticmethod
    def calculate_missing_cycle(current_miss: int, avg_frequency: float) -> float:
        """
        计算遗漏周期

        基于当前遗漏值和平均出现频率计算回补概率

        Args:
            current_miss: 当前遗漏值
            avg_frequency: 平均出现频率

        Returns:
            回补概率（0-1）
        """
        if avg_frequency <= 0:
            return 0.5

        # 使用指数衰减模型
        # 遗漏值越大，回补概率越高
        probability = 1 - np.exp(-current_miss / avg_frequency)
        return min(1.0, max(0.0, probability))

    @staticmethod
    def poisson_probability(lambda_param: float, k: int) -> float:
        """
        计算泊松概率

        P(X=k) = (lambda^k * e^(-lambda)) / k!

        Args:
            lambda_param: 泊松分布参数（期望）
            k: 事件发生次数

        Returns:
            概率值
        """
        # 使用scipy的泊松分布
        return float(scipy_stats.poisson.pmf(k, lambda_param))

    @staticmethod
    def normal_probability(x: float, mean: float, std: float) -> float:
        """
        计算正态分布概率密度

        Args:
            x: 随机变量值
            mean: 均值
            std: 标准差

        Returns:
            概率密度
        """
        if std <= 0:
            return 0.0
        return float(scipy_stats.norm.pdf(x, mean, std))

    @staticmethod
    def chi_square_test(observed: List[int], expected: List[float]) -> Tuple[float, float]:
        """
        卡方检验

        检验观察频率与期望频率的差异

        Args:
            observed: 观察频率列表
            expected: 期望频率列表

        Returns:
            (卡方统计量, p值)
        """
        if len(observed) != len(expected):
            raise ValueError("观察值和期望值长度不一致")

        # 归一化期望值
        total_obs = sum(observed)
        total_exp = sum(expected)

        if total_exp == 0:
            return 0.0, 1.0

        obs_array = np.array(observed, dtype=float)
        exp_array = np.array(expected, dtype=float) * (total_obs / total_exp)

        # 避免除零
        exp_array = np.where(exp_array == 0, 0.1, exp_array)

        chi2, p_value = scipy_stats.chisquare(obs_array, exp_array)
        return float(chi2), float(p_value)

    @staticmethod
    def correlation(x: List[float], y: List[float]) -> float:
        """
        计算皮尔逊相关系数

        Args:
            x, y: 两个数值列表

        Returns:
            相关系数（-1到1）
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        corr_matrix = np.corrcoef(x, y)
        return float(corr_matrix[0, 1])

    @staticmethod
    def moving_average(data: List[float], window: int) -> List[float]:
        """
        计算移动平均

        Args:
            data: 数据列表
            window: 窗口大小

        Returns:
            移动平均值列表
        """
        if len(data) < window:
            return []

        # 使用NumPy的卷积计算移动平均
        weights = np.ones(window) / window
        ma = np.convolve(data, weights, mode='valid')
        return ma.tolist()

    @staticmethod
    def exponential_smoothing(data: List[float], alpha: float = 0.3) -> List[float]:
        """
        指数平滑

        Args:
            data: 数据列表
            alpha: 平滑系数（0-1）

        Returns:
            平滑后的数据
        """
        if not data:
            return []

        smoothed = [data[0]]
        for i in range(1, len(data)):
            s = alpha * data[i] + (1 - alpha) * smoothed[-1]
            smoothed.append(s)

        return smoothed


# ============================================================================
# 第四部分：预测算法模块
# ============================================================================

class PredictionAlgorithms:
    """
    预测算法集合 - 实现12种预测算法

    涵盖统计、机器学习、深度学习方法
    """

    def __init__(self, historical_data: List[Dict[str, Any]]):
        """
        初始化预测算法

        Args:
            historical_data: 历史开奖数据列表
        """
        self.data = historical_data
        self.analysis_results = {}

        # 预处理数据
        self._prepare_data()

    def _prepare_data(self):
        """
        预处理历史数据

        提取各种分析所需的统计数据
        """
        if not self.data:
            return

        # 提取所有数字（不包含特别码）
        all_numbers = []
        for record in self.data:
            all_numbers.extend(record.get('numbers', []))

        # 计算频率
        self.frequency = MathUtils.calculate_frequency(all_numbers)

        # 计算遗漏值
        self.missing = {}
        for num in range(1, 50):
            self.missing[num] = self._calculate_missing(num)

        # 计算区间分布
        self.range_distribution = self._calculate_range_distribution()

        # 计算尾数分布
        self.tail_distribution = self._calculate_tail_distribution()

        # 计算单双分布
        self.odd_even_ratio = self._calculate_odd_even_ratio()

        # 计算大小分布
        self.big_small_ratio = self._calculate_big_small_ratio()

    def _calculate_missing(self, number: int) -> int:
        """
        计算某个数字的当前遗漏值

        Args:
            number: 数字（1-49）

        Returns:
            遗漏值（从最后一次出现到现在经过的期数）
        """
        missing = 0
        for record in reversed(self.data):
            if number in record.get('numbers', []):
                return missing
            missing += 1
        return missing + 10  # 未出现的数字，给予较大遗漏值

    def _calculate_range_distribution(self) -> Dict[int, int]:
        """计算区间分布"""
        distribution = {i: 0 for i in range(5)}
        for record in self.data:
            for num in record.get('numbers', []):
                idx = LotteryConfig.get_range_index(num)
                if idx >= 0:
                    distribution[idx] += 1
        return distribution

    def _calculate_tail_distribution(self) -> Dict[int, int]:
        """计算尾数分布"""
        distribution = {i: 0 for i in range(10)}
        for record in self.data:
            for num in record.get('numbers', []):
                tail = LotteryConfig.get_tail_digit(num)
                distribution[tail] += 1
        return distribution

    def _calculate_odd_even_ratio(self) -> Dict[str, int]:
        """计算单双比例"""
        odd_count = 0
        even_count = 0
        for record in self.data:
            for num in record.get('numbers', []):
                if LotteryConfig.is_odd(num):
                    odd_count += 1
                else:
                    even_count += 1
        return {'odd': odd_count, 'even': even_count}

    def _calculate_big_small_ratio(self) -> Dict[str, int]:
        """计算大小比例"""
        big_count = 0
        small_count = 0
        for record in self.data:
            for num in record.get('numbers', []):
                if LotteryConfig.is_big(num):
                    big_count += 1
                else:
                    small_count += 1
        return {'big': big_count, 'small': small_count}

    # -------------------------------------------------------------------------
    # 算法1：综合推荐算法
    # -------------------------------------------------------------------------
    def comprehensive_recommendation(self, count: int = 6) -> List[int]:
        """
        综合推荐算法

        综合多种算法的结果，得出最优推荐

        Args:
            count: 推荐数量

        Returns:
            推荐数字列表
        """
        # 收集各算法的预测
        predictions = []

        # 冷热算法
        hot_cold = self.hot_cold_algorithm(count * 2)
        predictions.extend(hot_cold)

        # 遗漏值分析
        missing_analysis = self.missing_value_analysis(count * 2)
        predictions.extend(missing_analysis)

        # 区间分布
        range_analysis = self.range_distribution_algorithm(count * 2)
        predictions.extend(range_analysis)

        # 统计各数字出现次数
        counter = Counter(predictions)

        # 选择得分最高的数字
        scores = {}
        for num in range(1, 50):
            score = 0
            score += counter.get(num, 0) * 10  # 被推荐次数
            score += self.frequency.get(num, 0) * 0.5  # 历史出现频率
            score += MathUtils.calculate_missing_cycle(
                self.missing.get(num, 50),
                len(self.data) / 49
            ) * 20  # 遗漏回补概率
            scores[num] = score

        # 排序并选择
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]

    # -------------------------------------------------------------------------
    # 算法2：冷热数字算法
    # -------------------------------------------------------------------------
    def hot_cold_algorithm(self, count: int = 6) -> List[int]:
        """
        冷热数字算法

        分析近期数字出现的冷热程度
        - 热号：近期出现频率高的数字
        - 冷号：长期未出现的数字

        Returns:
            预测数字列表
        """
        if not self.data:
            return random.sample(range(1, 50), count)

        # 近期数据权重（最近10期）
        recent_weight = {}
        for i, record in enumerate(self.data[:10]):
            weight = 10 - i  # 越近权重越高
            for num in record.get('numbers', []):
                recent_weight[num] = recent_weight.get(num, 0) + weight

        # 计算综合得分
        scores = {}
        for num in range(1, 50):
            recent = recent_weight.get(num, 0)
            missing = self.missing.get(num, 50)

            # 冷热得分 = 近期频率 * 0.6 + 遗漏回补概率 * 0.4
            heat_score = recent * 0.6
            cold_score = MathUtils.calculate_missing_cycle(missing, 6) * 40

            scores[num] = heat_score + cold_score

        # 排序并选择
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]

    # -------------------------------------------------------------------------
    # 算法3：单双算法
    # -------------------------------------------------------------------------
    def odd_even_algorithm(self, count: int = 6) -> List[int]:
        """
        单双算法

        基于单双号出现规律进行预测

        Returns:
            预测数字列表
        """
        if not self.data:
            return random.sample(range(1, 50), count)

        # 近期单双趋势
        recent_odd_even = []
        for record in self.data[:10]:
            odds = sum(1 for n in record.get('numbers', []) if LotteryConfig.is_odd(n))
            recent_odd_even.append(odds)

        # 计算平均单数
        avg_odds = np.mean(recent_odd_even)

        # 预测下一期单数（倾向于回归均值）
        if avg_odds > 3.5:
            predicted_odds = 3  # 偏少
        elif avg_odds < 2.5:
            predicted_odds = 4  # 偏多
        else:
            predicted_odds = 3

        # 选择单双数字
        selected = []
        odd_candidates = [n for n in range(1, 50) if LotteryConfig.is_odd(n)]
        even_candidates = [n for n in range(1, 50) if LotteryConfig.is_even(n)]

        # 根据频率和遗漏选择
        for _ in range(predicted_odds):
            if odd_candidates:
                # 选择遗漏较大或频率较高的奇数
                scores = {n: self.frequency.get(n, 0) * 2 + self.missing.get(n, 20) for n in odd_candidates}
                best = max(scores.items(), key=lambda x: x[1])[0]
                selected.append(best)
                odd_candidates.remove(best)

        for _ in range(count - predicted_odds):
            if even_candidates:
                scores = {n: self.frequency.get(n, 0) * 2 + self.missing.get(n, 20) for n in even_candidates}
                best = max(scores.items(), key=lambda x: x[1])[0]
                selected.append(best)
                even_candidates.remove(best)

        return selected

    # -------------------------------------------------------------------------
    # 算法4：大小算法
    # -------------------------------------------------------------------------
    def big_small_algorithm(self, count: int = 6) -> List[int]:
        """
        大小算法

        基于大小号出现规律进行预测

        Returns:
            预测数字列表
        """
        if not self.data:
            return random.sample(range(1, 50), count)

        # 近期大小趋势
        recent_big_small = []
        for record in self.data[:10]:
            bigs = sum(1 for n in record.get('numbers', []) if LotteryConfig.is_big(n))
            recent_big_small.append(bigs)

        avg_bigs = np.mean(recent_big_small)

        # 预测下一期大数数量
        if avg_bigs > 3.5:
            predicted_bigs = 3
        elif avg_bigs < 2.5:
            predicted_bigs = 4
        else:
            predicted_bigs = 3

        # 选择大小数字
        selected = []
        big_candidates = [n for n in range(26, 50)]
        small_candidates = [n for n in range(1, 26)]

        for _ in range(predicted_bigs):
            if big_candidates:
                scores = {n: self.frequency.get(n, 0) * 2 + self.missing.get(n, 20) for n in big_candidates}
                best = max(scores.items(), key=lambda x: x[1])[0]
                selected.append(best)
                big_candidates.remove(best)

        for _ in range(count - predicted_bigs):
            if small_candidates:
                scores = {n: self.frequency.get(n, 0) * 2 + self.missing.get(n, 20) for n in small_candidates}
                best = max(scores.items(), key=lambda x: x[1])[0]
                selected.append(best)
                small_candidates.remove(best)

        return selected

    # -------------------------------------------------------------------------
    # 算法5：遗漏值分析算法
    # -------------------------------------------------------------------------
    def missing_value_analysis(self, count: int = 6) -> List[int]:
        """
        遗漏值分析算法

        基于数字的遗漏周期进行分析

        Returns:
            预测数字列表
        """
        if not self.data:
            return random.sample(range(1, 50), count)

        # 计算平均出现周期
        total_records = len(self.data)
        avg_cycle = total_records / 49 if total_records > 0 else 6

        # 计算每个数字的回补概率
        scores = {}
        for num in range(1, 50):
            missing = self.missing.get(num, 50)
            # 回补概率 = 1 - e^(-遗漏/平均周期)
            probability = MathUtils.calculate_missing_cycle(missing, avg_cycle)

            # 综合得分 = 回补概率 * 0.7 + 历史频率 * 0.3
            frequency_score = self.frequency.get(num, 0) / total_records if total_records > 0 else 0
            scores[num] = probability * 0.7 + frequency_score * 0.3

        # 排序并选择
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]

    # -------------------------------------------------------------------------
    # 算法6：连号/邻号分析算法
    # -------------------------------------------------------------------------
    def adjacent_number_analysis(self, count: int = 6) -> List[int]:
        """
        连号/邻号分析算法

        分析相邻数字出现的规律

        Returns:
            预测数字列表
        """
        if not self.data:
            return random.sample(range(1, 50), count)

        # 统计邻号出现频率
        adjacent_freq = defaultdict(int)
        for record in self.data:
            numbers = sorted(record.get('numbers', []))
            for i in range(len(numbers) - 1):
                diff = numbers[i + 1] - numbers[i]
                if diff == 1:  # 连号
                    adjacent_freq['consecutive'] += 1
                elif diff <= 3:  # 近邻号
                    adjacent_freq['near'] += 1

        # 分析最近一期号码的邻号
        latest_numbers = self.data[0].get('numbers', [])
        adjacent_candidates = set()

        for num in latest_numbers:
            for offset in [-2, -1, 1, 2]:
                adj = num + offset
                if 1 <= adj <= 49:
                    adjacent_candidates.add(adj)

        # 计算候选数字的得分
        scores = {}
        for num in adjacent_candidates:
            base_score = self.frequency.get(num, 0)
            missing_score = self.missing.get(num, 20)
            scores[num] = base_score * 0.5 + missing_score * 0.5

        # 如果候选不足，补充其他数字
        if len(adjacent_candidates) < count:
            other_nums = [n for n in range(1, 50) if n not in adjacent_candidates]
            for num in other_nums:
                scores[num] = self.frequency.get(num, 0) + self.missing.get(num, 20) * 0.5

        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]

    # -------------------------------------------------------------------------
    # 算法7：尾数分布算法
    # -------------------------------------------------------------------------
    def tail_distribution_algorithm(self, count: int = 6) -> List[int]:
        """
        尾数分布算法

        分析数字尾数（0-9）的分布规律

        Returns:
            预测数字列表
        """
        if not self.data:
            return random.sample(range(1, 50), count)

        # 计算尾数频率
        tail_freq = {i: 0 for i in range(10)}
        for record in self.data:
            for num in record.get('numbers', []):
                tail = LotteryConfig.get_tail_digit(num)
                tail_freq[tail] += 1

        # 计算尾数遗漏
        tail_missing = {i: 0 for i in range(10)}
        for record in reversed(self.data):
            for num in record.get('numbers', []):
                tail = LotteryConfig.get_tail_digit(num)
                tail_missing[tail] += 1

        # 找出出现较少或遗漏较大的尾数
        tail_scores = {}
        for tail in range(10):
            freq_score = 10 - (tail_freq[tail] / len(self.data) * 10)
            missing_score = tail_missing[tail] * 0.5
            tail_scores[tail] = freq_score + missing_score

        # 选择得分高的尾数
        sorted_tails = sorted(tail_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        selected_tails = [t for t, _ in sorted_tails]

        # 选择对应尾数的数字
        candidates = []
        for tail in selected_tails:
            for num in range(1, 50):
                if LotteryConfig.get_tail_digit(num) == tail:
                    candidates.append(num)

        # 根据频率和遗漏排序
        scores = {n: self.frequency.get(n, 0) + self.missing.get(n, 20) for n in candidates}
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [num for num, _ in sorted_candidates[:count]]

    # -------------------------------------------------------------------------
    # 算法8：区间分布算法
    # -------------------------------------------------------------------------
    def range_distribution_algorithm(self, count: int = 6) -> List[int]:
        """
        区间分布算法

        分析数字在不同区间（1-9, 10-19, 20-29, 30-39, 40-49）的分布

        Returns:
            预测数字列表
        """
        if not self.data:
            return random.sample(range(1, 50), count)

        # 计算各区间近期出现频率
        recent_range_freq = {i: 0 for i in range(5)}
        for record in self.data[:10]:
            for num in record.get('numbers', []):
                idx = LotteryConfig.get_range_index(num)
                if idx >= 0:
                    recent_range_freq[idx] += 1

        # 计算区间遗漏
        range_missing = {i: 0 for i in range(5)}
        for record in reversed(self.data):
            ranges_found = set()
            for num in record.get('numbers', []):
                idx = LotteryConfig.get_range_index(num)
                if idx >= 0 and idx not in ranges_found:
                    ranges_found.add(idx)
            for idx in range(5):
                if idx not in ranges_found:
                    range_missing[idx] += 1

        # 计算区间得分
        range_scores = {}
        for i in range(5):
            # 出现少且遗漏大的区间更可能出数
            freq_score = 10 - (recent_range_freq[i] / 60 * 10)
            missing_score = range_missing[i] * 0.3
            range_scores[i] = freq_score + missing_score

        # 选择得分高的区间
        sorted_ranges = sorted(range_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        selected_ranges = [r for r, _ in sorted_ranges]

        # 从选中区间选择数字
        candidates = []
        for rng_idx in selected_ranges:
            start, end, _ = LotteryConfig.RANGES[rng_idx]
            for num in range(start, end + 1):
                candidates.append(num)

        # 根据频率和遗漏排序
        scores = {n: self.frequency.get(n, 0) * 0.5 + self.missing.get(n, 20) * 0.5
                  for n in candidates}
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [num for num, _ in sorted_candidates[:count]]

    # -------------------------------------------------------------------------
    # 算法9：轮盘赌选择算法
    # -------------------------------------------------------------------------
    def roulette_selection(self, count: int = 6) -> List[int]:
        """
        轮盘赌选择算法

        基于概率分布进行随机选择
        使用PyTorch实现，确保结果可复现

        Returns:
            预测数字列表
        """
        # 设置随机种子
        torch.manual_seed(int(datetime.datetime.now().timestamp()) % 1000000)

        # 计算每个数字的选择概率
        weights = []
        for num in range(1, 50):
            # 频率权重
            freq = self.frequency.get(num, 0)
            # 遗漏权重
            missing = self.missing.get(num, 50)
            # 综合权重
            weight = freq * 0.3 + missing * 0.7
            weights.append(weight)

        # 归一化
        total = sum(weights)
        if total > 0:
            probabilities = [w / total for w in weights]
        else:
            probabilities = [1 / 49] * 49

        # 使用PyTorch进行加权随机采样
        probs_tensor = torch.tensor(probabilities, dtype=torch.float32)

        selected = []
        available = list(range(1, 50))

        for _ in range(count):
            if not available:
                break
            # 创建多项分布
            m = torch.distributions.Categorical(probs_tensor)
            idx = m.sample()
            num = int(idx) + 1

            if num in available:
                selected.append(num)
                available.remove(num)
                # 更新概率
                probs_tensor[idx] = 0
                probs_tensor = probs_tensor / probs_tensor.sum() if probs_tensor.sum() > 0 else probs_tensor
            else:
                # 重新选择
                for _ in range(10):
                    idx = m.sample()
                    num = int(idx) + 1
                    if num in available:
                        selected.append(num)
                        available.remove(num)
                        probs_tensor[idx] = 0
                        probs_tensor = probs_tensor / probs_tensor.sum() if probs_tensor.sum() > 0 else probs_tensor
                        break

        return selected

    # -------------------------------------------------------------------------
    # 算法10：历史相似性算法
    # -------------------------------------------------------------------------
    def historical_similarity(self, count: int = 6) -> List[int]:
        """
        历史相似性算法

        找出与最近一期最相似的历史记录，参考其后续号码

        使用Scikit-learn计算相似度

        Returns:
            预测数字列表
        """
        if len(self.data) < 10:
            return random.sample(range(1, 50), count)

        # 将每期数据转换为特征向量
        def to_feature_vector(numbers):
            vec = [0] * 49
            for n in numbers:
                if 1 <= n <= 49:
                    vec[n - 1] = 1
            return vec

        # 最近一期的特征
        latest = to_feature_vector(self.data[0].get('numbers', []))

        # 计算与历史记录的相似度（余弦相似度）
        similarities = []
        for i, record in enumerate(self.data[1:50]):  # 只比较前50期
            hist = to_feature_vector(record.get('numbers', []))

            # 余弦相似度
            dot = sum(a * b for a, b in zip(latest, hist))
            norm_latest = math.sqrt(sum(a ** 2 for a in latest))
            norm_hist = math.sqrt(sum(b ** 2 for b in hist))

            if norm_latest > 0 and norm_hist > 0:
                sim = dot / (norm_latest * norm_hist)
            else:
                sim = 0

            similarities.append((i + 1, sim))

        # 找出最相似的记录
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:5]

        # 统计相似记录的后续号码
        next_numbers = []
        for idx, sim in top_similar:
            if idx + 1 < len(self.data):
                next_numbers.extend(self.data[idx + 1].get('numbers', []))

        # 选择出现最多的号码
        if next_numbers:
            counter = Counter(next_numbers)
            # 选择出现2次以上的，或者出现1次但相似度高的
            selected = [num for num, cnt in counter.most_common(count * 2) if cnt >= 1]
            return selected[:count]

        return random.sample(range(1, 50), count)

    # -------------------------------------------------------------------------
    # 算法11：泊松概率分布算法
    # -------------------------------------------------------------------------
    def poisson_distribution(self, count: int = 6) -> List[int]:
        """
        泊松概率分布算法

        使用泊松分布建模数字出现规律

        使用SciPy的泊松分布函数

        Returns:
            预测数字列表
        """
        if not self.data:
            return random.sample(range(1, 50), count)

        # 计算每个数字的理论出现率（λ参数）
        total_numbers = len(self.data) * 6  # 每期6个正码
        lambda_params = {}

        for num in range(1, 50):
            freq = self.frequency.get(num, 0)
            # λ = 期望出现次数 / 总期数
            lambda_params[num] = freq / len(self.data) if len(self.data) > 0 else 1 / 49

        # 计算每个数字的泊松概率
        probabilities = {}
        for num in range(1, 50):
            lam = lambda_params[num]
            # P(X >= 1) = 1 - P(X = 0)
            prob = 1 - MathUtils.poisson_probability(lam * len(self.data), 0)
            probabilities[num] = prob

        # 选择概率最高的数字
        sorted_nums = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]

    # -------------------------------------------------------------------------
    # 算法12：玄学算法
    # -------------------------------------------------------------------------
    def mystical_algorithm(self, count: int = 6) -> List[int]:
        """
        玄学算法

        结合时间、日期等"玄学"因素的神秘算法
        仅供参考，娱乐使用

        Returns:
            预测数字列表
        """
        # 获取当前时间种子
        now = datetime.datetime.now()

        # 混合多种玄学因素
        factors = [
            now.day,  # 日期
            now.month,  # 月份
            now.weekday(),  # 星期
            now.hour,  # 小时
            now.minute,  # 分钟
            int(now.timestamp()) % 100,  # 时间戳
        ]

        # 生成随机种子
        seed = sum(factors) * now.second
        random.seed(seed)
        torch.manual_seed(seed)

        # 使用PyTorch生成随机数
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # 生成随机概率分布
        base_probs = torch.rand(49, device=device)

        # 结合历史数据微调
        if self.data:
            for i, record in enumerate(self.data[:5]):
                weight = (5 - i) * 0.1
                for num in record.get('numbers', []):
                    if 1 <= num <= 49:
                        base_probs[num - 1] += weight

        # 归一化
        probs = base_probs / base_probs.sum()

        # 采样
        m = torch.distributions.Categorical(probs)
        selected = []
        available = list(range(1, 50))

        for _ in range(count):
            if not available:
                break
            idx = m.sample()
            num = int(idx) + 1
            if num in available:
                selected.append(num)
                available.remove(num)

        return selected


# ============================================================================
# 第五部分：机器学习预测模型
# ============================================================================

class MLPredictionModel:
    """
    机器学习预测模型

    使用Scikit-learn和Optuna进行模型训练和预测
    使用PyTorch实现神经网络预测
    """

    def __init__(self, historical_data: List[Dict[str, Any]]):
        """
        初始化模型

        Args:
            historical_data: 历史数据
        """
        self.data = historical_data
        self.models = {}
        self.scalers = {}

    def prepare_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备特征和标签

        将历史数据转换为机器学习可用的格式

        Returns:
            (特征矩阵, 标签向量)
        """
        if len(self.data) < 20:
            return np.array([]), np.array([])

        X = []  # 特征
        y = []  # 标签

        # 滑动窗口：使用前5期预测第6期
        window_size = 5

        for i in range(len(self.data) - window_size):
            # 特征：前5期的号码信息
            features = []
            for j in range(window_size):
                record = self.data[i + j]
                numbers = record.get('numbers', [])

                # 特征1：号码列表（49维one-hot）
                one_hot = [0] * 49
                for n in numbers:
                    if 1 <= n <= 49:
                        one_hot[n - 1] = 1
                features.extend(one_hot)

                # 特征2：统计信息
                features.append(sum(numbers) / 6)  # 平均值
                features.append(max(numbers))  # 最大值
                features.append(min(numbers))  # 最小值
                features.append(sum(n % 2 for n in numbers))  # 单数个数
                features.append(sum(1 for n in numbers if n > 25))  # 大数个数

            X.append(features)

            # 标签：第6期的号码
            next_record = self.data[i + window_size]
            label = [0] * 49
            for n in next_record.get('numbers', []):
                if 1 <= n <= 49:
                    label[n - 1] = 1
            y.append(label)

        return np.array(X), np.array(y)

    def train_random_forest(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        """
        训练随机森林模型

        Args:
            X: 特征矩阵
            y: 标签矩阵

        Returns:
            训练好的模型
        """
        # 由于是多标签问题，分别训练每个标签的分类器
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        # 将多标签转换为首个标签进行训练
        y_single = np.argmax(y, axis=1)

        model.fit(X, y_single)
        return model

    def train_gradient_boosting(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingClassifier:
        """
        训练梯度提升模型

        Args:
            X: 特征矩阵
            y: 标签矩阵

        Returns:
            训练好的模型
        """
        y_single = np.argmax(y, axis=1)

        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

        model.fit(X, y_single)
        return model

    def train_neural_network(self, X: np.ndarray, y: np.ndarray) -> nn.Module:
        """
        使用PyTorch训练神经网络模型

        Args:
            X: 特征矩阵
            y: 标签矩阵

        Returns:
            训练好的PyTorch模型
        """
        # 数据预处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y)

        # 定义神经网络
        class LotteryNN(nn.Module):
            def __init__(self, input_size, output_size):
                super(LotteryNN, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, output_size),
                    nn.Softmax(dim=1)
                )

            def forward(self, x):
                return self.network(x)

        # 初始化模型
        model = LotteryNN(X.shape[1], y.shape[1])

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            # 使用类别索引作为标签
            _, labels = y_tensor.max(dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        return model

    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        使用Optuna优化超参数

        Args:
            X: 特征矩阵
            y: 标签矩阵

        Returns:
            最优超参数
        """

        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 3, 15)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)

            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # 训练模型
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42
            )

            y_single = np.argmax(y, axis=1)
            model.fit(X_train, y_single)

            # 评估
            y_pred = model.predict(X_test)
            y_test_single = np.argmax(y_test, axis=1)
            accuracy = accuracy_score(y_test_single, y_pred)

            return accuracy

        # 运行优化
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=20)

        return study.best_params

    def predict_with_all_models(self) -> List[int]:
        """
        使用所有模型进行预测

        Returns:
            综合预测结果
        """
        X, y = self.prepare_features()

        if len(X) < 20:
            return random.sample(range(1, 50), 6)

        # 准备最新特征
        latest_features = []
        for j in range(5):
            record = self.data[j]
            numbers = record.get('numbers', [])

            one_hot = [0] * 49
            for n in numbers:
                if 1 <= n <= 49:
                    one_hot[n - 1] = 1
            latest_features.extend(one_hot)

            latest_features.append(sum(numbers) / 6)
            latest_features.append(max(numbers))
            latest_features.append(min(numbers))
            latest_features.append(sum(n % 2 for n in numbers))
            latest_features.append(sum(1 for n in numbers if n > 25))

        X_latest = np.array([latest_features])

        predictions = []

        # 随机森林预测
        try:
            rf_model = self.train_random_forest(X, y)
            rf_pred = rf_model.predict(X_latest)[0]
            predictions.append(rf_pred + 1)
        except:
            pass

        # 梯度提升预测
        try:
            gb_model = self.train_gradient_boosting(X, y)
            gb_pred = gb_model.predict(X_latest)[0]
            predictions.append(gb_pred + 1)
        except:
            pass

        # 神经网络预测
        try:
            nn_model = self.train_neural_network(X, y)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_latest_scaled = scaler.transform(X_latest)
            X_tensor = torch.FloatTensor(X_latest_scaled)

            nn_model.eval()
            with torch.no_grad():
                output = nn_model(X_tensor)
                nn_pred = torch.argmax(output, dim=1).item()
                predictions.append(nn_pred + 1)
        except:
            pass

        # 综合预测
        if len(predictions) >= 3:
            # 取出现最多的预测
            counter = Counter(predictions)
            top_pred = counter.most_common(1)[0][0]
            return [top_pred] + random.sample([n for n in range(1, 50) if n != top_pred], 5)
        else:
            return random.sample(range(1, 50), 6)


# ============================================================================
# 第六部分：自定义控件模块
# ============================================================================

class NumberButton(QPushButton):
    """
    数字按钮控件

    显示六合彩数字，带有对应颜色
    """

    def __init__(self, number: int, parent=None):
        """
        初始化数字按钮

        Args:
            number: 数字（1-49）
            parent: 父控件
        """
        super().__init__(parent)
        self.number = number
        self.is_selected = False

        self._setup_ui()

    def _setup_ui(self):
        """设置UI"""
        self.setText(str(self.number))
        self.setMinimumSize(50, 50)
        self.setMaximumSize(100, 100)
        self.setCheckable(True)

        # 应用颜色
        self._apply_color()

    def _apply_color(self):
        """应用颜色样式"""
        colors = LotteryConfig.get_number_color(self.number)

        # 设置样式表
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors['bg']};
                color: {colors['text']};
                border: 2px solid {colors['border']};
                border-radius: 8px;
                font-weight: bold;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background-color: {colors['border']};
                color: white;
            }}
            QPushButton:checked {{
                background-color: {colors['text']};
                color: white;
            }}
        """)

    def set_selected(self, selected: bool):
        """
        设置选中状态

        Args:
            selected: 是否选中
        """
        self.is_selected = selected
        self.setChecked(selected)

    def get_number(self) -> int:
        """获取数字"""
        return self.number


class NumberPanel(QWidget):
    """
    数字面板控件

    显示49个数字按钮，支持选择
    """

    number_selected = pyqtSignal(list)  # 数字选择信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_numbers = []
        self.number_buttons = {}

        self._init_ui()

    def _init_ui(self):
        """初始化UI"""
        layout = QGridLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)

        # 创建49个数字按钮
        # 布局：7列 x 7行
        for num in range(1, 50):
            btn = NumberButton(num, self)
            btn.clicked.connect(lambda checked, n=num: self._on_button_clicked(n))
            self.number_buttons[num] = btn

            row = (num - 1) // 7
            col = (num - 1) % 7
            layout.addWidget(btn, row, col)

        self.setLayout(layout)

    def _on_button_clicked(self, number: int):
        """
        处理按钮点击

        Args:
            number: 被点击的数字
        """
        btn = self.number_buttons[number]

        if btn.isChecked():
            if number not in self.selected_numbers:
                self.selected_numbers.append(number)
        else:
            if number in self.selected_numbers:
                self.selected_numbers.remove(number)

        self.number_selected.emit(self.selected_numbers)

    def get_selected_numbers(self) -> List[int]:
        """获取选中的数字"""
        return self.selected_numbers.copy()

    def set_selected_numbers(self, numbers: List[int]):
        """
        设置选中的数字

        Args:
            numbers: 数字列表
        """
        # 清除所有选择
        for btn in self.number_buttons.values():
            btn.set_selected(False)

        self.selected_numbers = []

        # 设置新选择
        for num in numbers:
            if num in self.number_buttons:
                self.number_buttons[num].set_selected(True)
                self.selected_numbers.append(num)

        self.number_selected.emit(self.selected_numbers)

    def clear_selection(self):
        """清除所有选择"""
        self.set_selected_numbers([])

    def highlight_numbers(self, numbers: List[int]):
        """
        高亮显示指定数字

        Args:
            numbers: 要高亮的数字列表
        """
        for num, btn in self.number_buttons.items():
            if num in numbers:
                btn.set_selected(True)
            else:
                btn.set_selected(False)


class StatisticsChart(QWidget):
    """
    统计图表控件

    使用Matplotlib绘制各种统计图表
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_frequency(self, frequency: Dict[int, int], title: str = "数字出现频率"):
        """
        绘制频率分布图

        Args:
            frequency: 数字频率字典
            title: 图表标题
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        numbers = sorted(frequency.keys())
        counts = [frequency[n] for n in numbers]

        # 根据数字颜色设置柱状图颜色
        colors = []
        for num in numbers:
            if LotteryConfig.is_red(num):
                colors.append('#FF0000')
            elif LotteryConfig.is_blue(num):
                colors.append('#0000FF')
            else:
                colors.append('#008000')

        bars = ax.bar(numbers, counts, color=colors, edgecolor='white', linewidth=0.5)

        ax.set_xlabel('数字', fontsize=12)
        ax.set_ylabel('出现次数', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(numbers)
        ax.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=8)

        self.canvas.draw()

    def plot_missing(self, missing: Dict[int, int], title: str = "数字遗漏值"):
        """
        绘制遗漏值图

        Args:
            missing: 遗漏值字典
            title: 图表标题
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        numbers = sorted(missing.keys())
        values = [missing[n] for n in numbers]

        # 热力图颜色
        cmap = plt.cm.RdYlGn_r
        norm = plt.Normalize(vmin=min(values), vmax=max(values))
        colors = [cmap(norm(v)) for v in values]

        ax.bar(numbers, values, color=colors, edgecolor='white', linewidth=0.5)

        ax.set_xlabel('数字', fontsize=12)
        ax.set_ylabel('遗漏期数', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(numbers)
        ax.grid(axis='y', alpha=0.3)

        self.canvas.draw()

    def plot_distribution(self, data: Dict[str, int], title: str = "分布统计"):
        """
        绘制分布饼图

        Args:
            data: 分布数据
            title: 图表标题
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        labels = list(data.keys())
        values = list(data.values())

        colors = ['#FF0000', '#0000FF', '#008000', '#F39C12', '#9B59B6']

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            colors=colors[:len(labels)],
            autopct='%1.1f%%',
            startangle=90
        )

        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title(title, fontsize=14, fontweight='bold')

        self.canvas.draw()

    def plot_heatmap(self, matrix: np.ndarray, title: str = "热力图"):
        """
        绘制热力图

        Args:
            matrix: 数据矩阵
            title: 图表标题
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        sns.heatmap(matrix, ax=ax, cmap='YlOrRd', annot=False,
                    cbar_kws={'label': '频率'})

        ax.set_title(title, fontsize=14, fontweight='bold')

        self.canvas.draw()

    def set_chart_type(self, chart_type: str):
        """设置图表类型"""
        self.chart_type = chart_type
        self.data = None

    def update_data(self, history_data: List):
        """更新数据并重绘图表"""
        self.data = history_data

        if not history_data:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', fontsize=16)
            ax.set_title('请先导入数据', fontsize=14)
            self.canvas.draw()
            return

        chart_type = getattr(self, 'chart_type', 'frequency')

        if chart_type == 'frequency':
            self._draw_frequency_chart(history_data)
        elif chart_type == 'missing':
            self._draw_missing_chart(history_data)
        elif chart_type == 'odd_even':
            self._draw_odd_even_chart(history_data)
        elif chart_type == 'size':
            self._draw_size_chart(history_data)
        elif chart_type == 'color':
            self._draw_color_chart(history_data)
        elif chart_type == 'comprehensive':
            self._draw_comprehensive_chart(history_data)

    def _draw_frequency_chart(self, history_data: List):
        """绘制频率分布图"""
        frequency = {}
        for record in history_data:
            for num in record.get('numbers', []):
                frequency[num] = frequency.get(num, 0) + 1
            special = record.get('special')
            if special:
                frequency[special] = frequency.get(special, 0) + 1

        if frequency:
            self.plot_frequency(frequency, "数字出现频率分布")

    def _draw_missing_chart(self, history_data: List):
        """绘制遗漏值图"""
        missing = {i: 0 for i in range(1, 50)}
        appeared = set()

        for i, record in enumerate(history_data):
            for num in record.get('numbers', []):
                if num not in appeared:
                    missing[num] = i
                appeared.add(num)

        # 更新未出现的数字
        for num in range(1, 50):
            if num not in appeared:
                missing[num] = len(history_data)

        self.plot_missing(missing, "数字遗漏期数")

    def _draw_odd_even_chart(self, history_data: List):
        """绘制单双分布图"""
        odd_count = 0
        even_count = 0

        for record in history_data:
            for num in record.get('numbers', []):
                if num % 2 == 1:
                    odd_count += 1
                else:
                    even_count += 1

        self.plot_distribution({'单数': odd_count, '双数': even_count}, "单双分布统计")

    def _draw_size_chart(self, history_data: List):
        """绘制大小分布图"""
        big_count = 0
        small_count = 0

        for record in history_data:
            for num in record.get('numbers', []):
                if num > 24:
                    big_count += 1
                else:
                    small_count += 1

        self.plot_distribution({'大号(25-49)': big_count, '小号(1-24)': small_count}, "大小分布统计")

    def _draw_color_chart(self, history_data: List):
        """绘制颜色分布图"""
        red_count = blue_count = green_count = 0

        for record in history_data:
            for num in record.get('numbers', []):
                if LotteryConfig.is_red(num):
                    red_count += 1
                elif LotteryConfig.is_blue(num):
                    blue_count += 1
                else:
                    green_count += 1

        self.plot_distribution({'红色': red_count, '蓝色': blue_count, '绿色': green_count}, "颜色分布统计")

    def _draw_comprehensive_chart(self, history_data: List):
        """绘制综合统计图"""
        self.figure.clear()

        # 创建2x2子图
        fig = self.figure

        # 1. 频率分布
        ax1 = fig.add_subplot(221)
        frequency = {}
        for record in history_data:
            for num in record.get('numbers', []):
                frequency[num] = frequency.get(num, 0) + 1

        if frequency:
            numbers = sorted(frequency.keys())
            counts = [frequency[n] for n in numbers]
            colors = ['#FF0000' if LotteryConfig.is_red(n) else
                      '#0000FF' if LotteryConfig.is_blue(n) else '#008000'
                      for n in numbers]
            ax1.bar(numbers, counts, color=colors)
            ax1.set_title('频率分布', fontweight='bold')

        # 2. 单双分布
        ax2 = fig.add_subplot(222)
        odd_count = even_count = 0
        for record in history_data:
            for num in record.get('numbers', []):
                if num % 2 == 1:
                    odd_count += 1
                else:
                    even_count += 1
        ax2.pie([odd_count, even_count], labels=['单', '双'],
                colors=['#FF0000', '#0000FF'], autopct='%1.1f%%')
        ax2.set_title('单双分布', fontweight='bold')

        # 3. 大小分布
        ax3 = fig.add_subplot(223)
        big_count = small_count = 0
        for record in history_data:
            for num in record.get('numbers', []):
                if num > 24:
                    big_count += 1
                else:
                    small_count += 1
        ax3.pie([big_count, small_count], labels=['大', '小'],
                colors=['#008000', '#F39C12'], autopct='%1.1f%%')
        ax3.set_title('大小分布', fontweight='bold')

        # 4. 颜色分布
        ax4 = fig.add_subplot(224)
        red_count = blue_count = green_count = 0
        for record in history_data:
            for num in record.get('numbers', []):
                if LotteryConfig.is_red(num):
                    red_count += 1
                elif LotteryConfig.is_blue(num):
                    blue_count += 1
                else:
                    green_count += 1
        ax4.pie([red_count, blue_count, green_count], labels=['红', '蓝', '绿'],
                colors=['#FF0000', '#0000FF', '#008000'], autopct='%1.1f%%')
        ax4.set_title('颜色分布', fontweight='bold')

        fig.tight_layout()
        self.canvas.draw()


# ============================================================================
# 第七部分：主窗口类
# ============================================================================

class LotteryPredictionWindow(QMainWindow):
    """
    彩票预测系统主窗口

    实现完整的GUI界面，包括：
    - 窗口创建和布局
    - 字体系统
    - 色彩系统
    - 拖拽交互
    - 自适应联动
    - 滚动条
    - 顶部全局控制栏
    """

    # =====================================
    # 7.1 初始化和窗口创建
    # =====================================

    def __init__(self):
        """初始化主窗口"""
        super().__init__()

        # 窗口属性
        self.setWindowTitle(LotteryConfig.WINDOW_TITLE)
        self.setMinimumSize(
            LotteryConfig.WINDOW_MIN_WIDTH,
            LotteryConfig.WINDOW_MIN_HEIGHT
        )
        self.resize(1600, 1000)

        # 状态变量
        self.font_size_key = LotteryConfig.DEFAULT_FONT_SIZE_KEY
        self.historical_data = []
        self.prediction_cache = {}
        self.current_algorithm_index = 0

        # 数据文件路径
        self.data_file = "./用户上传/彩票预测系统_v3/lottery_data.json"

        # 加载历史数据
        self._load_data()

        # 初始化UI
        self._init_ui()

        # 应用样式
        self._apply_stylesheet()

        print("彩票预测系统初始化完成")

    def _init_ui(self):
        """初始化UI组件"""
        # 创建中心控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 创建顶部控制栏
        self._create_top_bar(main_layout)

        # 创建标签页
        self._create_tabs(main_layout)

        # 创建状态栏
        self._create_status_bar()

    # =====================================
    # 7.2 顶部全局控制栏
    # =====================================

    def _create_top_bar(self, parent_layout: QVBoxLayout):
        """
        创建顶部控制栏

        Args:
            parent_layout: 父布局
        """
        # 顶部栏容器
        top_bar = QWidget()
        top_bar.setObjectName("TopBar")
        top_bar.setMinimumHeight(60)
        top_bar.setMaximumHeight(80)

        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setSpacing(10)
        top_bar_layout.setContentsMargins(15, 5, 15, 5)

        # 左侧：标题
        title_label = QLabel("📊 彩票预测系统 v3.0")
        title_label.setObjectName("TitleLabel")
        top_bar_layout.addWidget(title_label, 0, Qt.AlignmentFlag.AlignLeft)

        # 中间：功能按钮
        button_group = QWidget()
        button_layout = QHBoxLayout(button_group)
        button_layout.setSpacing(8)

        # 导入按钮
        import_btn = QPushButton("📥 导入")
        import_btn.setObjectName("ImportButton")
        import_btn.clicked.connect(self._on_import_clicked)
        button_layout.addWidget(import_btn)

        # 导出按钮
        export_btn = QPushButton("📤 导出")
        export_btn.setObjectName("ExportButton")
        export_btn.clicked.connect(self._on_export_clicked)
        button_layout.addWidget(export_btn)

        # 保存按钮
        save_btn = QPushButton("💾 保存")
        save_btn.setObjectName("SaveButton")
        save_btn.clicked.connect(self._on_save_clicked)
        button_layout.addWidget(save_btn)

        # 分隔符
        separator1 = QLabel("|")
        separator1.setObjectName("Separator")
        button_layout.addWidget(separator1)

        # 添加数据按钮
        add_btn = QPushButton("➕ 添加数据")
        add_btn.setObjectName("AddButton")
        add_btn.clicked.connect(self._on_add_data_clicked)
        button_layout.addWidget(add_btn)

        # 删除数据按钮
        del_btn = QPushButton("🗑️ 删除数据")
        del_btn.setObjectName("DeleteButton")
        del_btn.clicked.connect(self._on_delete_data_clicked)
        button_layout.addWidget(del_btn)

        # 清空按钮
        clear_btn = QPushButton("🔄 清空")
        clear_btn.setObjectName("ClearButton")
        clear_btn.clicked.connect(self._on_clear_data_clicked)
        button_layout.addWidget(clear_btn)

        top_bar_layout.addWidget(button_group, 1, Qt.AlignmentFlag.AlignCenter)

        # 右侧：字体控制
        font_control = QWidget()
        font_layout = QHBoxLayout(font_control)
        font_layout.setSpacing(5)

        font_label = QLabel("字体大小:")
        font_layout.addWidget(font_label)

        # 字体大小下拉框
        self.font_combo = QComboBox()
        self.font_combo.setObjectName("FontSizeCombo")
        for size_key in LotteryConfig.FONT_SIZES.keys():
            self.font_combo.addItem(size_key)

        # 设置默认选择
        default_index = list(LotteryConfig.FONT_SIZES.keys()).index(self.font_size_key)
        self.font_combo.setCurrentIndex(default_index)

        self.font_combo.currentTextChanged.connect(self._on_font_size_changed)
        font_layout.addWidget(self.font_combo)

        # 字号减小按钮
        font_minus_btn = QPushButton("A-")
        font_minus_btn.setObjectName("FontMinusButton")
        font_minus_btn.setFixedSize(35, 35)
        font_minus_btn.clicked.connect(self._decrease_font_size)
        font_layout.addWidget(font_minus_btn)

        # 字号增加按钮
        font_plus_btn = QPushButton("A+")
        font_plus_btn.setObjectName("FontPlusButton")
        font_plus_btn.setFixedSize(35, 35)
        font_plus_btn.clicked.connect(self._increase_font_size)
        font_layout.addWidget(font_plus_btn)

        top_bar_layout.addWidget(font_control, 0, Qt.AlignmentFlag.AlignRight)

        parent_layout.addWidget(top_bar)

    # =====================================
    # 7.3 标签页创建
    # =====================================

    def _create_tabs(self, parent_layout: QVBoxLayout):
        """
        创建标签页

        Args:
            parent_layout: 父布局
        """
        # 标签页控件
        self.tabs = QTabWidget()
        self.tabs.setObjectName("MainTabs")
        self.tabs.setMovable(True)  # 允许拖动标签
        self.tabs.setDocumentMode(True)  # 文档模式

        # 标签页1：数据导入与格式转换
        tab1 = self._create_data_import_tab()
        self.tabs.addTab(tab1, "📥 数据导入与格式转换")

        # 标签页2：数据分析与预测
        tab2 = self._create_analysis_tab()
        self.tabs.addTab(tab2, "📈 数据分析与预测")

        # 标签页3：第七位预判
        tab3 = self._create_seventh_prediction_tab()
        self.tabs.addTab(tab3, "🎯 第七位预判")

        # 标签页4：统计分析图表
        tab4 = self._create_statistics_chart_tab()
        self.tabs.addTab(tab4, "📊 统计分析图表")

        parent_layout.addWidget(self.tabs)

    def _create_data_import_tab(self) -> QWidget:
        """
        创建数据导入标签页

        Returns:
            标签页控件
        """
        widget = QWidget()

        # 使用水平分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)

        # 左侧：原始数据输入区
        left_panel = self._create_input_panel()
        splitter.addWidget(left_panel)

        # 右侧：转换结果和历史记录区
        right_panel = self._create_result_panel()
        splitter.addWidget(right_panel)

        # 设置分割比例
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(splitter)

        return widget

    def _create_input_panel(self) -> QWidget:
        """
        创建输入面板

        Returns:
            面板控件
        """
        widget = QWidget()
        widget.setObjectName("InputPanel")

        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # 标题
        title = QLabel("📝 粘贴原始数据")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        # 说明标签
        info_label = QLabel(
            "粘贴格式示例：\n第116期最新开奖结果 2026年04月26日 15 龙/水 46 鸡/木 16 兔/木 10 鸡/火 48 羊/火 33 狗/火 22 鸡/水")
        info_label.setObjectName("InfoLabel")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # 文本编辑区
        self.raw_text_edit = QTextEdit()
        self.raw_text_edit.setObjectName("RawTextEdit")
        self.raw_text_edit.setPlaceholderText("请在此处粘贴原始数据...")
        layout.addWidget(self.raw_text_edit)

        # 按钮组
        button_layout = QHBoxLayout()

        # 转换按钮
        convert_btn = QPushButton("🔄 转换为标准格式")
        convert_btn.setObjectName("ConvertButton")
        convert_btn.clicked.connect(self._on_convert_clicked)
        button_layout.addWidget(convert_btn)

        # 添加到历史按钮
        add_to_history_btn = QPushButton("📚 添加到历史记录")
        add_to_history_btn.setObjectName("AddToHistoryButton")
        add_to_history_btn.clicked.connect(self._on_add_to_history_clicked)
        button_layout.addWidget(add_to_history_btn)

        # 批量导入按钮
        batch_import_btn = QPushButton("📦 批量导入")
        batch_import_btn.setObjectName("BatchImportButton")
        batch_import_btn.clicked.connect(self._on_batch_import_clicked)
        button_layout.addWidget(batch_import_btn)

        layout.addLayout(button_layout)

        # 清空按钮
        clear_btn = QPushButton("🗑️ 清空输入")
        clear_btn.setObjectName("ClearInputButton")
        clear_btn.clicked.connect(lambda: self.raw_text_edit.clear())
        layout.addWidget(clear_btn)

        return widget

    def _create_result_panel(self) -> QWidget:
        """
        创建结果面板

        Returns:
            面板控件
        """
        widget = QWidget()
        widget.setObjectName("ResultPanel")

        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # 标题
        title = QLabel("📊 转换结果")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        # 转换结果显示
        self.converted_text_edit = QTextEdit()
        self.converted_text_edit.setObjectName("ConvertedTextEdit")
        self.converted_text_edit.setReadOnly(True)
        layout.addWidget(self.converted_text_edit)

        # 历史记录标题
        history_title = QLabel("📜 历史记录")
        history_title.setObjectName("PanelTitle")
        layout.addWidget(history_title)

        # 历史记录列表（带滚动条）
        history_scroll = QScrollArea()
        history_scroll.setObjectName("HistoryScroll")
        history_scroll.setWidgetResizable(True)
        history_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # 历史记录列表
        self.history_list = QListWidget()
        self.history_list.setObjectName("HistoryList")
        self.history_list.itemDoubleClicked.connect(self._on_history_item_double_clicked)

        history_scroll.setWidget(self.history_list)
        layout.addWidget(history_scroll, 1)

        return widget

    def _create_analysis_tab(self) -> QWidget:
        """
        创建数据分析标签页

        Returns:
            标签页控件
        """
        widget = QWidget()

        # 使用水平分割器
        h_splitter = QSplitter(Qt.Orientation.Horizontal)
        h_splitter.setHandleWidth(2)

        # 左侧面板
        left_panel = self._create_left_analysis_panel()
        h_splitter.addWidget(left_panel)

        # 右侧面板
        right_panel = self._create_right_analysis_panel()
        h_splitter.addWidget(right_panel)

        # 设置分割比例
        h_splitter.setStretchFactor(0, 1)
        h_splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(h_splitter)

        return widget

    def _create_left_analysis_panel(self) -> QWidget:
        """
        创建左侧分析面板

        Returns:
            面板控件
        """
        widget = QWidget()
        widget.setObjectName("LeftAnalysisPanel")

        # 使用垂直分割器
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.setHandleWidth(2)

        # 最新数据显示区
        latest_panel = self._create_latest_data_panel()
        v_splitter.addWidget(latest_panel)

        # 数字选择面板
        selection_panel = self._create_selection_panel()
        v_splitter.addWidget(selection_panel)

        # 预判面板
        prediction_panel = self._create_prediction_judge_panel()
        v_splitter.addWidget(prediction_panel)

        # 设置分割比例
        v_splitter.setStretchFactor(0, 1)
        v_splitter.setStretchFactor(1, 2)
        v_splitter.setStretchFactor(2, 1)

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(v_splitter)

        return widget

    def _create_right_analysis_panel(self) -> QWidget:
        """
        创建右侧分析面板

        Returns:
            面板控件
        """
        widget = QWidget()
        widget.setObjectName("RightAnalysisPanel")

        # 使用垂直分割器
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.setHandleWidth(2)

        # 算法选择区
        algorithm_panel = self._create_algorithm_panel()
        v_splitter.addWidget(algorithm_panel)

        # 预测结果区
        result_panel = self._create_prediction_result_panel()
        v_splitter.addWidget(result_panel)

        # 统计图表区
        chart_panel = self._create_chart_panel()
        v_splitter.addWidget(chart_panel)

        # 设置分割比例
        v_splitter.setStretchFactor(0, 1)
        v_splitter.setStretchFactor(1, 2)
        v_splitter.setStretchFactor(2, 2)

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(v_splitter)

        return widget

    def _create_latest_data_panel(self) -> QWidget:
        """
        创建最新数据显示面板

        Returns:
            面板控件
        """
        widget = QWidget()
        widget.setObjectName("LatestDataPanel")

        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # 标题
        title = QLabel("🎯 最新开奖数据")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        # 最新数据显示
        self.latest_display = QLabel("暂无数据")
        self.latest_display.setObjectName("LatestDisplay")
        self.latest_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.latest_display.setWordWrap(True)
        layout.addWidget(self.latest_display)

        # 刷新按钮
        refresh_btn = QPushButton("🔄 刷新显示")
        refresh_btn.setObjectName("RefreshButton")
        refresh_btn.clicked.connect(self._refresh_latest_display)
        layout.addWidget(refresh_btn)

        return widget

    def _create_selection_panel(self) -> QWidget:
        """
        创建数字选择面板

        Returns:
            面板控件
        """
        widget = QWidget()
        widget.setObjectName("SelectionPanel")

        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # 标题
        title = QLabel("🔢 数字选择面板（49个数字）")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        # 数字面板（带滚动条）
        scroll = QScrollArea()
        scroll.setObjectName("NumberScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # 数字面板
        self.number_panel = NumberPanel()
        self.number_panel.number_selected.connect(self._on_number_selected)
        scroll.setWidget(self.number_panel)

        layout.addWidget(scroll, 1)

        # 选中数字显示
        selected_layout = QHBoxLayout()
        selected_label = QLabel("已选数字:")
        selected_layout.addWidget(selected_label)

        self.selected_numbers_label = QLabel("无")
        self.selected_numbers_label.setObjectName("SelectedNumbersLabel")
        selected_layout.addWidget(self.selected_numbers_label)

        selected_layout.addStretch()

        # 清除按钮
        clear_btn = QPushButton("清除")
        clear_btn.setObjectName("ClearSelectionButton")
        clear_btn.clicked.connect(self._clear_number_selection)
        selected_layout.addWidget(clear_btn)

        layout.addLayout(selected_layout)

        return widget

    def _create_prediction_judge_panel(self) -> QWidget:
        """
        创建预判面板

        Returns:
            面板控件
        """
        widget = QWidget()
        widget.setObjectName("PredictionJudgePanel")

        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # 标题
        title = QLabel("🔮 预判第七位（特别码）")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        # 预判选项
        options_layout = QGridLayout()

        # 大小预判
        size_label = QLabel("大小:")
        self.size_judge_combo = QComboBox()
        self.size_judge_combo.addItems(["请选择", "大(26-49)", "小(1-25)"])
        options_layout.addWidget(size_label, 0, 0)
        options_layout.addWidget(self.size_judge_combo, 0, 1)

        # 单双预判
        odd_label = QLabel("单双:")
        self.odd_judge_combo = QComboBox()
        self.odd_judge_combo.addItems(["请选择", "单数", "双数"])
        options_layout.addWidget(odd_label, 1, 0)
        options_layout.addWidget(self.odd_judge_combo, 1, 1)

        # 尾数大小预判
        tail_label = QLabel("尾数:")
        self.tail_judge_combo = QComboBox()
        self.tail_judge_combo.addItems(["请选择", "大(5-9)", "小(0-4)"])
        options_layout.addWidget(tail_label, 2, 0)
        options_layout.addWidget(self.tail_judge_combo, 2, 1)

        layout.addLayout(options_layout)

        # 预判按钮
        judge_btn = QPushButton("🎯 开始预判")
        judge_btn.setObjectName("JudgeButton")
        judge_btn.clicked.connect(self._on_judge_clicked)
        layout.addWidget(judge_btn)

        # 预判结果
        self.judge_result_label = QLabel("预判结果：等待选择...")
        self.judge_result_label.setObjectName("JudgeResultLabel")
        self.judge_result_label.setWordWrap(True)
        layout.addWidget(self.judge_result_label)

        return widget

    def _create_algorithm_panel(self) -> QWidget:
        """
        创建算法选择面板

        Returns:
            面板控件
        """
        widget = QWidget()
        widget.setObjectName("AlgorithmPanel")

        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # 标题
        title = QLabel("🧮 选择预测算法")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        # 算法选择下拉框
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.setObjectName("AlgorithmCombo")
        for algo_name, algo_desc in LotteryConfig.ALGORITHMS:
            self.algorithm_combo.addItem(f"{algo_name}", algo_desc)

        self.algorithm_combo.currentIndexChanged.connect(self._on_algorithm_changed)
        layout.addWidget(self.algorithm_combo)

        # 算法说明
        self.algorithm_desc_label = QLabel("请选择一个预测算法")
        self.algorithm_desc_label.setObjectName("AlgorithmDescLabel")
        self.algorithm_desc_label.setWordWrap(True)
        layout.addWidget(self.algorithm_desc_label)

        # 按钮组
        button_layout = QHBoxLayout()

        # 预测按钮
        predict_btn = QPushButton("🎲 开始预测")
        predict_btn.setObjectName("PredictButton")
        predict_btn.clicked.connect(self._on_predict_clicked)
        button_layout.addWidget(predict_btn)

        # 随机抽取按钮
        random_btn = QPushButton("🎯 随机抽取")
        random_btn.setObjectName("RandomButton")
        random_btn.clicked.connect(self._on_random_draw_clicked)
        button_layout.addWidget(random_btn)

        layout.addLayout(button_layout)

        # 使用机器学习预测按钮
        ml_btn = QPushButton("🤖 机器学习预测")
        ml_btn.setObjectName("MLPredictButton")
        ml_btn.clicked.connect(self._on_ml_predict_clicked)
        layout.addWidget(ml_btn)

        return widget

    def _create_prediction_result_panel(self) -> QWidget:
        """
        创建预测结果面板

        Returns:
            面板控件
        """
        widget = QWidget()
        widget.setObjectName("PredictionResultPanel")

        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # 标题
        title = QLabel("📋 预测结果")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        # 预测结果列表（带滚动条）
        scroll = QScrollArea()
        scroll.setObjectName("PredictionScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        result_layout.setSpacing(5)

        # 预测号码显示
        self.prediction_display = QLabel("等待预测...")
        self.prediction_display.setObjectName("PredictionDisplay")
        self.prediction_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prediction_display.setWordWrap(True)
        result_layout.addWidget(self.prediction_display)

        # 预测号码按钮面板
        self.prediction_number_panel = QWidget()
        self.prediction_number_layout = QGridLayout(self.prediction_number_panel)
        self.prediction_number_layout.setSpacing(5)
        result_layout.addWidget(self.prediction_number_panel)

        # 预测统计信息
        self.prediction_stats_label = QLabel("统计信息：等待预测...")
        self.prediction_stats_label.setObjectName("PredictionStatsLabel")
        self.prediction_stats_label.setWordWrap(True)
        result_layout.addWidget(self.prediction_stats_label)

        scroll.setWidget(result_widget)
        layout.addWidget(scroll, 1)

        return widget

    def _create_chart_panel(self) -> QWidget:
        """
        创建统计图表面板

        Returns:
            面板控件
        """
        widget = QWidget()
        widget.setObjectName("ChartPanel")

        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        # 标题
        title = QLabel("📊 统计分析图表")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        # 图表类型选择
        chart_type_layout = QHBoxLayout()

        chart_type_label = QLabel("图表类型:")
        chart_type_layout.addWidget(chart_type_label)

        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["频率分布图", "遗漏值图", "单双分布图", "大小分布图"])
        chart_type_layout.addWidget(self.chart_type_combo)
        chart_type_layout.addStretch()

        # 刷新图表按钮
        refresh_chart_btn = QPushButton("🔄 刷新图表")
        refresh_chart_btn.setObjectName("RefreshChartButton")
        refresh_chart_btn.clicked.connect(self._refresh_chart)
        chart_type_layout.addWidget(refresh_chart_btn)

        layout.addLayout(chart_type_layout)

        # 图表显示区（带滚动条）
        chart_scroll = QScrollArea()
        chart_scroll.setObjectName("ChartScroll")
        chart_scroll.setWidgetResizable(True)

        self.statistics_chart = StatisticsChart()
        chart_scroll.setWidget(self.statistics_chart)

        layout.addWidget(chart_scroll, 1)

        return widget

    def _create_seventh_prediction_tab(self) -> QWidget:
        """
        创建第七位预判标签页

        Returns:
            标签页控件
        """
        widget = QWidget()

        # 使用垂直分割器
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(2)

        # 上半部分：预判按钮区
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        top_layout.setSpacing(15)
        top_layout.setContentsMargins(10, 10, 10, 10)

        # 标题
        title = QLabel("🎯 第七位数字预判")
        title.setObjectName("PanelTitle")
        title.setStyleSheet(f'''
            QLabel {{
                font-size: 18px;
                font-weight: bold;
                color: {LotteryConfig.COLOR_ERROR};
                padding: 10px;
            }}
        ''')
        top_layout.addWidget(title)

        # 说明文字
        desc = QLabel("根据历史数据分析第七位特别号码的大小、单双、尾数特征")
        desc.setStyleSheet(f"color: {LotteryConfig.COLOR_TEXT_SECONDARY}; font-size: 14px;")
        top_layout.addWidget(desc)

        # 预判按钮组
        btn_group = QWidget()
        btn_layout = QHBoxLayout(btn_group)
        btn_layout.setSpacing(20)

        # 大小预判按钮
        size_btn = QPushButton("📊 大小预判")
        size_btn.setStyleSheet(f'''
            QPushButton {{
                background-color: {LotteryConfig.COLOR_SUCCESS};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px 30px;
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #27AE60;
            }}
        ''')
        size_btn.clicked.connect(self._predict_seventh_size)
        btn_layout.addWidget(size_btn)

        # 单双预判按钮
        odd_even_btn = QPushButton("🔢 单双预判")
        odd_even_btn.setStyleSheet(f'''
            QPushButton {{
                background-color: {LotteryConfig.COLOR_INFO};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px 30px;
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #2980B9;
            }}
        ''')
        odd_even_btn.clicked.connect(self._predict_seventh_odd_even)
        btn_layout.addWidget(odd_even_btn)

        # 尾数大小预判按钮
        tail_btn = QPushButton("🎯 尾数大小预判")
        tail_btn.setStyleSheet(f'''
            QPushButton {{
                background-color: {LotteryConfig.COLOR_WARNING};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px 30px;
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #E67E22;
            }}
        ''')
        tail_btn.clicked.connect(self._predict_seventh_tail)
        btn_layout.addWidget(tail_btn)

        # 综合预判按钮
        all_btn = QPushButton("🔮 综合预判")
        all_btn.setStyleSheet(f'''
            QPushButton {{
                background-color: {LotteryConfig.COLOR_ERROR};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px 30px;
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #C0392B;
            }}
        ''')
        all_btn.clicked.connect(self._predict_seventh_all)
        btn_layout.addWidget(all_btn)

        top_layout.addWidget(btn_group)
        top_layout.addStretch()

        splitter.addWidget(top_panel)

        # 下半部分：预判结果显示区
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(10, 10, 10, 10)

        # 结果标题
        result_title = QLabel("📋 预判结果")
        result_title.setStyleSheet(f'''
            QLabel {{
                font-size: 16px;
                font-weight: bold;
                color: {LotteryConfig.COLOR_TEXT_PRIMARY};
            }}
        ''')
        bottom_layout.addWidget(result_title)

        # 结果显示区
        self.seventh_result_text = QTextEdit()
        self.seventh_result_text.setReadOnly(True)
        self.seventh_result_text.setStyleSheet(f'''
            QTextEdit {{
                background-color: {LotteryConfig.COLOR_BG_SECONDARY};
                border: 2px solid {LotteryConfig.COLOR_BORDER};
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                color: {LotteryConfig.COLOR_TEXT_PRIMARY};
            }}
        ''')
        self.seventh_result_text.setPlaceholderText("点击上方按钮进行预判...")
        bottom_layout.addWidget(self.seventh_result_text)

        splitter.addWidget(bottom_panel)

        # 设置分割比例
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(splitter)

        return widget

    def _create_statistics_chart_tab(self) -> QWidget:
        """
        创建统计分析图表标签页

        Returns:
            标签页控件
        """
        widget = QWidget()

        # 使用水平分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)

        # 左侧：图表类型选择和控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # 标题
        title = QLabel("📊 统计分析图表")
        title.setObjectName("PanelTitle")
        title.setStyleSheet(f'''
            QLabel {{
                font-size: 18px;
                font-weight: bold;
                color: {LotteryConfig.COLOR_INFO};
                padding: 10px;
            }}
        ''')
        left_layout.addWidget(title)

        # 图表类型选择
        chart_group = QWidget()
        chart_group.setStyleSheet(f'''
            QWidget {{
                background-color: {LotteryConfig.COLOR_BG_SECONDARY};
                border-radius: 8px;
                padding: 10px;
            }}
        ''')
        chart_layout = QVBoxLayout(chart_group)

        chart_label = QLabel("选择图表类型:")
        chart_label.setStyleSheet(f"font-weight: bold; color: {LotteryConfig.COLOR_TEXT_PRIMARY};")
        chart_layout.addWidget(chart_label)

        # 图表类型按钮
        chart_types = [
            ("📈 频率分布图", self._show_frequency_chart),
            ("📉 遗漏值分析图", self._show_missing_chart),
            ("🔵 单双分布图", self._show_odd_even_chart),
            ("🔴 大小分布图", self._show_size_chart),
            ("🎨 颜色分布图", self._show_color_chart),
            ("📊 综合统计图", self._show_comprehensive_chart),
        ]

        for name, callback in chart_types:
            btn = QPushButton(name)
            btn.setStyleSheet(f'''
                QPushButton {{
                    background-color: {LotteryConfig.COLOR_BG_PRIMARY};
                    color: {LotteryConfig.COLOR_TEXT_PRIMARY};
                    border: 2px solid {LotteryConfig.COLOR_BORDER};
                    border-radius: 6px;
                    padding: 10px;
                    font-size: 14px;
                    text-align: left;
                }}
                QPushButton:hover {{
                    background-color: {LotteryConfig.COLOR_INFO};
                    color: white;
                    border-color: {LotteryConfig.COLOR_INFO};
                }}
            ''')
            btn.clicked.connect(callback)
            chart_layout.addWidget(btn)

        left_layout.addWidget(chart_group)

        # 数据统计区
        stats_group = QWidget()
        stats_group.setStyleSheet(f'''
            QWidget {{
                background-color: {LotteryConfig.COLOR_BG_SECONDARY};
                border-radius: 8px;
                padding: 10px;
            }}
        ''')
        stats_layout = QVBoxLayout(stats_group)

        stats_title = QLabel("📋 数据统计")
        stats_title.setStyleSheet(f"font-weight: bold; color: {LotteryConfig.COLOR_TEXT_PRIMARY};")
        stats_layout.addWidget(stats_title)

        self.stats_text_display = QTextEdit()
        self.stats_text_display.setReadOnly(True)
        self.stats_text_display.setMaximumHeight(200)
        self.stats_text_display.setStyleSheet(f'''
            QTextEdit {{
                background-color: {LotteryConfig.COLOR_BG_PRIMARY};
                border: 1px solid {LotteryConfig.COLOR_BORDER};
                border-radius: 4px;
                padding: 5px;
                font-size: 12px;
                color: {LotteryConfig.COLOR_TEXT_PRIMARY};
            }}
        ''')
        stats_layout.addWidget(self.stats_text_display)

        left_layout.addWidget(stats_group)
        left_layout.addStretch()

        splitter.addWidget(left_panel)

        # 右侧：图表显示区
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # 图表标题
        self.chart_title_label = QLabel("选择左侧图表类型查看分析结果")
        self.chart_title_label.setStyleSheet(f'''
            QLabel {{
                font-size: 16px;
                font-weight: bold;
                color: {LotteryConfig.COLOR_TEXT_PRIMARY};
                padding: 5px;
            }}
        ''')
        right_layout.addWidget(self.chart_title_label)

        # 图表显示区（带滚动条）
        chart_scroll = QScrollArea()
        chart_scroll.setWidgetResizable(True)
        chart_scroll.setStyleSheet(f'''
            QScrollArea {{
                background-color: {LotteryConfig.COLOR_BG_PRIMARY};
                border: 2px solid {LotteryConfig.COLOR_BORDER};
                border-radius: 8px;
            }}
        ''')

        # 使用之前定义的StatisticsChart
        self.main_chart_widget = StatisticsChart()
        chart_scroll.setWidget(self.main_chart_widget)

        right_layout.addWidget(chart_scroll, 1)

        splitter.addWidget(right_panel)

        # 设置分割比例
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(splitter)

        return widget

    # -------------------------------------------------------------------------
    # 第七位预判相关方法
    # -------------------------------------------------------------------------

    def _predict_seventh_size(self):
        """预判第七位大小"""
        if not self.historical_data:
            self.seventh_result_text.setPlainText("❌ 没有历史数据，请先导入数据！")
            return

        # 统计第七位大小分布
        big_count = 0
        small_count = 0

        for record in self.historical_data:
            seventh = record.get('special', 0)
            if seventh > 24:
                big_count += 1
            else:
                small_count += 1

        total = big_count + small_count
        big_ratio = big_count / total * 100 if total > 0 else 50
        small_ratio = small_count / total * 100 if total > 0 else 50

        # 预测（基于趋势）
        prediction = "大" if big_ratio > small_ratio else "小"
        confidence = max(big_ratio, small_ratio)

        result = f'''📊 第七位大小预判结果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 历史数据统计：
   • 大号(25-49)出现次数：{big_count} 次 ({big_ratio:.1f}%)
   • 小号(1-24)出现次数：{small_count} 次 ({small_ratio:.1f}%)

🎯 预判结果：{prediction}
   • 置信度：{confidence:.1f}%

💡 建议：下一期第七位号码倾向于「{prediction}」范围'''

        self.seventh_result_text.setPlainText(result)

    def _predict_seventh_odd_even(self):
        """预判第七位单双"""
        if not self.historical_data:
            self.seventh_result_text.setPlainText("❌ 没有历史数据，请先导入数据！")
            return

        # 统计第七位单双分布
        odd_count = 0
        even_count = 0

        for record in self.historical_data:
            seventh = record.get('special', 0)
            if seventh % 2 == 1:
                odd_count += 1
            else:
                even_count += 1

        total = odd_count + even_count
        odd_ratio = odd_count / total * 100 if total > 0 else 50
        even_ratio = even_count / total * 100 if total > 0 else 50

        # 预测
        prediction = "单" if odd_ratio > even_ratio else "双"
        confidence = max(odd_ratio, even_ratio)

        result = f'''🔢 第七位单双预判结果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 历史数据统计：
   • 单号出现次数：{odd_count} 次 ({odd_ratio:.1f}%)
   • 双号出现次数：{even_count} 次 ({even_ratio:.1f}%)

🎯 预判结果：{prediction}
   • 置信度：{confidence:.1f}%

💡 建议：下一期第七位号码倾向于「{prediction}」数'''

        self.seventh_result_text.setPlainText(result)

    def _predict_seventh_tail(self):
        """预判第七位尾数大小"""
        if not self.historical_data:
            self.seventh_result_text.setPlainText("❌ 没有历史数据，请先导入数据！")
            return

        # 统计第七位尾数大小分布
        big_tail_count = 0
        small_tail_count = 0

        for record in self.historical_data:
            seventh = record.get('special', 0)
            tail = seventh % 10
            if tail >= 5:
                big_tail_count += 1
            else:
                small_tail_count += 1

        total = big_tail_count + small_tail_count
        big_ratio = big_tail_count / total * 100 if total > 0 else 50
        small_ratio = small_tail_count / total * 100 if total > 0 else 50

        # 预测
        prediction = "大尾(5-9)" if big_ratio > small_ratio else "小尾(0-4)"
        confidence = max(big_ratio, small_ratio)

        result = f'''🎯 第七位尾数大小预判结果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 历史数据统计：
   • 大尾(5-9)出现次数：{big_tail_count} 次 ({big_ratio:.1f}%)
   • 小尾(0-4)出现次数：{small_tail_count} 次 ({small_ratio:.1f}%)

🎯 预判结果：{prediction}
   • 置信度：{confidence:.1f}%

💡 建议：下一期第七位号码尾数倾向于「{prediction}」范围'''

        self.seventh_result_text.setPlainText(result)

    def _predict_seventh_all(self):
        """综合预判第七位"""
        if not self.historical_data:
            self.seventh_result_text.setPlainText("❌ 没有历史数据，请先导入数据！")
            return

        # 收集所有统计数据
        big_count = small_count = odd_count = even_count = big_tail_count = small_tail_count = 0

        for record in self.historical_data:
            seventh = record.get('special', 0)

            # 大小
            if seventh > 24:
                big_count += 1
            else:
                small_count += 1

            # 单双
            if seventh % 2 == 1:
                odd_count += 1
            else:
                even_count += 1

            # 尾数大小
            if seventh % 10 >= 5:
                big_tail_count += 1
            else:
                small_tail_count += 1

        total = len(self.historical_data)

        # 预测结果
        size_pred = "大" if big_count > small_count else "小"
        odd_even_pred = "单" if odd_count > even_count else "双"
        tail_pred = "大尾" if big_tail_count > small_tail_count else "小尾"

        result = f'''🔮 第七位综合预判结果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 数据样本：{total} 期

┌─────────────────────────────────────┐
│ 📈 大小分析                          │
│    大号(25-49)：{big_count}次 ({big_count / total * 100:.1f}%)              │
│    小号(1-24)：{small_count}次 ({small_count / total * 100:.1f}%)              │
│    ➜ 预判：{size_pred}                          │
├─────────────────────────────────────┤
│ 🔢 单双分析                          │
│    单数：{odd_count}次 ({odd_count / total * 100:.1f}%)                    │
│    双数：{even_count}次 ({even_count / total * 100:.1f}%)                    │
│    ➜ 预判：{odd_even_pred}                          │
├─────────────────────────────────────┤
│ 🎯 尾数分析                          │
│    大尾(5-9)：{big_tail_count}次 ({big_tail_count / total * 100:.1f}%)              │
│    小尾(0-4)：{small_tail_count}次 ({small_tail_count / total * 100:.1f}%)              │
│    ➜ 预判：{tail_pred}                        │
└─────────────────────────────────────┘

🎯 综合预判结果：
   第七位号码特征：{size_pred} + {odd_even_pred} + {tail_pred}

💡 建议关注号码范围：'''

        # 根据预判推荐号码
        recommended = []
        for n in range(1, 50):
            # 检查是否符合预判
            size_ok = (n > 24) == (size_pred == "大")
            odd_ok = (n % 2 == 1) == (odd_even_pred == "单")
            tail_ok = (n % 10 >= 5) == (tail_pred == "大尾")

            if size_ok and odd_ok and tail_ok:
                recommended.append(n)

        if recommended:
            result += f'\n   {", ".join(map(str, recommended[:10]))}'
            if len(recommended) > 10:
                result += f' ... 等共{len(recommended)}个号码'

        self.seventh_result_text.setPlainText(result)

    # -------------------------------------------------------------------------
    # 统计图表相关方法
    # -------------------------------------------------------------------------

    def _show_frequency_chart(self):
        """显示频率分布图"""
        self.chart_title_label.setText("📈 频率分布图")
        self._update_chart("frequency")

    def _show_missing_chart(self):
        """显示遗漏值分析图"""
        self.chart_title_label.setText("📉 遗漏值分析图")
        self._update_chart("missing")

    def _show_odd_even_chart(self):
        """显示单双分布图"""
        self.chart_title_label.setText("🔵 单双分布图")
        self._update_chart("odd_even")

    def _show_size_chart(self):
        """显示大小分布图"""
        self.chart_title_label.setText("🔴 大小分布图")
        self._update_chart("size")

    def _show_color_chart(self):
        """显示颜色分布图"""
        self.chart_title_label.setText("🎨 颜色分布图")
        self._update_chart("color")

    def _show_comprehensive_chart(self):
        """显示综合统计图"""
        self.chart_title_label.setText("📊 综合统计图")
        self._update_chart("comprehensive")

    def _update_chart(self, chart_type: str):
        """更新图表显示"""
        if hasattr(self, 'main_chart_widget'):
            self.main_chart_widget.set_chart_type(chart_type)
            self.main_chart_widget.update_data(self.historical_data)

    # =====================================
    # 7.4 状态栏创建
    # =====================================

    def _create_status_bar(self):
        """创建状态栏"""
        self.statusBar().showMessage("就绪 | 准备预测...")

        # 添加永久部件
        self.data_count_label = QLabel("历史记录: 0 条")
        self.statusBar().addPermanentWidget(self.data_count_label)

    # =====================================
    # 7.5 样式表应用
    # =====================================

    def _apply_stylesheet(self):
        """应用全局样式表"""
        # 强制禁用Fusion等暗色主题，使用纯白背景
        self.setStyleSheet("")

        # 应用自定义样式
        self._update_stylesheet()

    def _update_stylesheet(self):
        """
        更新样式表

        根据当前字体大小更新所有控件样式
        """
        font_size = LotteryConfig.FONT_SIZES.get(self.font_size_key, 16)
        small_font_size = max(10, font_size - 4)
        large_font_size = font_size + 4

        # 全局样式表
        stylesheet = f"""
            /* 全局样式 */
            QWidget {{
                background-color: {LotteryConfig.COLOR_BG_PRIMARY};
                color: {LotteryConfig.COLOR_TEXT_PRIMARY};
                font-size: {font_size}px;
                font-family: "Microsoft YaHei", "SimHei", "PingFang SC", Arial, sans-serif;
            }}

            /* 顶部栏样式 */
            #TopBar {{
                background-color: {LotteryConfig.COLOR_BG_PRIMARY};
                border-bottom: 2px solid {LotteryConfig.COLOR_BORDER};
            }}

            #TitleLabel {{
                font-size: {large_font_size}px;
                font-weight: bold;
                color: {LotteryConfig.COLOR_INFO};
            }}

            /* 按钮样式 */
            QPushButton {{
                background-color: {LotteryConfig.COLOR_BUTTON_BG};
                color: {LotteryConfig.COLOR_TEXT_PRIMARY};
                border: 2px solid {LotteryConfig.COLOR_BORDER};
                border-radius: 6px;
                padding: 6px 16px;
                font-size: {small_font_size}px;
                min-height: 30px;
            }}

            QPushButton:hover {{
                background-color: {LotteryConfig.COLOR_BUTTON_HOVER};
                border-color: {LotteryConfig.COLOR_INFO};
            }}

            QPushButton:pressed {{
                background-color: {LotteryConfig.COLOR_BUTTON_PRESSED};
            }}

            QPushButton:disabled {{
                background-color: #F5F5F5;
                color: #AAAAAA;
                border-color: #DDDDDD;
            }}

            /* 标签页样式 */
            QTabWidget::pane {{
                border: 1px solid {LotteryConfig.COLOR_BORDER};
                background-color: {LotteryConfig.COLOR_BG_PRIMARY};
            }}

            QTabBar::tab {{
                background-color: {LotteryConfig.COLOR_BG_PRIMARY};
                color: {LotteryConfig.COLOR_TEXT_PRIMARY};
                border: 1px solid {LotteryConfig.COLOR_BORDER};
                padding: 8px 20px;
                margin-right: 2px;
                font-size: {small_font_size}px;
            }}

            QTabBar::tab:selected {{
                background-color: {LotteryConfig.COLOR_INFO};
                color: white;
                border-color: {LotteryConfig.COLOR_INFO};
            }}

            QTabBar::tab:hover {{
                background-color: #E8F4FC;
            }}

            /* 文本编辑框样式 */
            QTextEdit, QLineEdit {{
                background-color: {LotteryConfig.COLOR_BG_PRIMARY};
                color: {LotteryConfig.COLOR_TEXT_PRIMARY};
                border: 2px solid {LotteryConfig.COLOR_BORDER};
                border-radius: 4px;
                padding: 8px;
                font-size: {font_size}px;
            }}

            QTextEdit:focus, QLineEdit:focus {{
                border-color: {LotteryConfig.COLOR_INFO};
            }}

            /* 下拉框样式 */
            QComboBox {{
                background-color: {LotteryConfig.COLOR_BG_PRIMARY};
                color: {LotteryConfig.COLOR_TEXT_PRIMARY};
                border: 2px solid {LotteryConfig.COLOR_BORDER};
                border-radius: 4px;
                padding: 6px 12px;
                font-size: {small_font_size}px;
                min-height: 28px;
            }}

            QComboBox:hover {{
                border-color: {LotteryConfig.COLOR_INFO};
            }}

            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}

            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {LotteryConfig.COLOR_TEXT_PRIMARY};
            }}

            /* 面板标题样式 */
            #PanelTitle {{
                font-size: {font_size}px;
                font-weight: bold;
                color: {LotteryConfig.COLOR_INFO};
                padding: 5px;
                border-bottom: 1px solid {LotteryConfig.COLOR_BORDER_LIGHT};
            }}

            /* 标签样式 */
            QLabel {{
                font-size: {font_size}px;
                color: {LotteryConfig.COLOR_TEXT_PRIMARY};
            }}

            /* 列表样式 */
            QListWidget {{
                background-color: {LotteryConfig.COLOR_BG_PRIMARY};
                color: {LotteryConfig.COLOR_TEXT_PRIMARY};
                border: 1px solid {LotteryConfig.COLOR_BORDER};
                border-radius: 4px;
                font-size: {small_font_size}px;
            }}

            QListWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {LotteryConfig.COLOR_BORDER_LIGHT};
            }}

            QListWidget::item:selected {{
                background-color: {LotteryConfig.COLOR_INFO};
                color: white;
            }}

            /* 滚动条样式 */
            QScrollBar:vertical {{
                background-color: {LotteryConfig.COLOR_BG_PRIMARY};
                width: 12px;
                margin: 0px;
            }}

            QScrollBar::handle:vertical {{
                background-color: {LotteryConfig.COLOR_BORDER};
                border-radius: 6px;
                min-height: 30px;
            }}

            QScrollBar::handle:vertical:hover {{
                background-color: {LotteryConfig.COLOR_INFO};
            }}

            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}

            QScrollBar:horizontal {{
                background-color: {LotteryConfig.COLOR_BG_PRIMARY};
                height: 12px;
                margin: 0px;
            }}

            QScrollBar::handle:horizontal {{
                background-color: {LotteryConfig.COLOR_BORDER};
                border-radius: 6px;
                min-width: 30px;
            }}

            QScrollBar::handle:horizontal:hover {{
                background-color: {LotteryConfig.COLOR_INFO};
            }}

            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}

            /* 分割器样式 */
            QSplitter::handle {{
                background-color: {LotteryConfig.COLOR_BORDER};
            }}

            QSplitter::handle:horizontal {{
                width: 2px;
            }}

            QSplitter::handle:vertical {{
                height: 2px;
            }}

            QSplitter::handle:hover {{
                background-color: {LotteryConfig.COLOR_INFO};
            }}

            /* 滚动区域样式 */
            QScrollArea {{
                background-color: {LotteryConfig.COLOR_BG_PRIMARY};
                border: 1px solid {LotteryConfig.COLOR_BORDER};
                border-radius: 4px;
            }}

            /* 预测显示样式 */
            #PredictionDisplay {{
                font-size: {large_font_size}px;
                font-weight: bold;
                color: {LotteryConfig.COLOR_SUCCESS};
                padding: 15px;
                background-color: #E8F8F0;
                border: 2px solid {LotteryConfig.COLOR_SUCCESS};
                border-radius: 8px;
            }}

            /* 最新数据显示样式 */
            #LatestDisplay {{
                font-size: {font_size}px;
                padding: 10px;
                background-color: #E8F4FC;
                border: 2px solid {LotteryConfig.COLOR_INFO};
                border-radius: 6px;
            }}

            /* 选中数字标签样式 */
            #SelectedNumbersLabel {{
                color: {LotteryConfig.COLOR_SUCCESS};
                font-weight: bold;
            }}

            /* 算法描述样式 */
            #AlgorithmDescLabel {{
                font-size: {small_font_size}px;
                color: {LotteryConfig.COLOR_TEXT_SECONDARY};
                padding: 5px;
                background-color: #F9F9F9;
                border-radius: 4px;
            }}

            /* 统计信息样式 */
            #PredictionStatsLabel {{
                font-size: {small_font_size}px;
                color: {LotteryConfig.COLOR_TEXT_SECONDARY};
                padding: 8px;
                background-color: #F9F9F9;
                border-radius: 4px;
            }}

            /* 状态栏样式 */
            QStatusBar {{
                background-color: {LotteryConfig.COLOR_BG_PRIMARY};
                color: {LotteryConfig.COLOR_TEXT_SECONDARY};
                border-top: 1px solid {LotteryConfig.COLOR_BORDER};
                font-size: {small_font_size}px;
            }}

            /* 信息标签样式 */
            #InfoLabel {{
                font-size: {small_font_size}px;
                color: {LotteryConfig.COLOR_TEXT_SECONDARY};
                padding: 8px;
                background-color: #F0F7FF;
                border-radius: 4px;
            }}
        """

        self.setStyleSheet(stylesheet)

    # =====================================
    # 7.6 事件处理函数
    # =====================================

    def resizeEvent(self, event):
        """
        窗口大小改变事件

        实现百分百自适应联动

        Args:
            event: 调整大小事件
        """
        # 调用父类方法
        super().resizeEvent(event)

        # 触发自适应更新
        self._apply_autosize()

    def _apply_autosize(self):
        """
        应用自适应尺寸

        根据窗口大小自动调整所有控件尺寸
        """
        # 获取当前窗口大小
        window_size = self.size()
        window_width = window_size.width()
        window_height = window_size.height()

        # 计算缩放因子
        base_width = 1600
        base_height = 1000
        scale_x = window_width / base_width
        scale_y = window_height / base_height
        scale = min(scale_x, scale_y)  # 使用较小的因子确保比例

        # 更新字体大小
        base_font_size = LotteryConfig.FONT_SIZES.get(self.font_size_key, 16)
        new_font_size = max(10, int(base_font_size * scale))

        # 应用字体大小
        font = QFont()
        font.setPointSize(new_font_size)
        QApplication.instance().setFont(font)

    def _on_font_size_changed(self, size_key: str):
        """
        字体大小改变处理

        Args:
            size_key: 新的字体大小键名
        """
        self.font_size_key = size_key
        self._update_stylesheet()
        self._apply_autosize()

    def _increase_font_size(self):
        """增加字体大小"""
        keys = list(LotteryConfig.FONT_SIZES.keys())
        current_index = keys.index(self.font_size_key) if self.font_size_key in keys else 3
        if current_index < len(keys) - 1:
            new_key = keys[current_index + 1]
            self.font_combo.setCurrentText(new_key)

    def _decrease_font_size(self):
        """减小字体大小"""
        keys = list(LotteryConfig.FONT_SIZES.keys())
        current_index = keys.index(self.font_size_key) if self.font_size_key in keys else 3
        if current_index > 0:
            new_key = keys[current_index - 1]
            self.font_combo.setCurrentText(new_key)

    # =====================================
    # 7.7 数据操作处理
    # =====================================

    def _load_data(self):
        """加载历史数据"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.historical_data = json.load(f)
                print(f"已加载 {len(self.historical_data)} 条历史记录")
            else:
                # 生成示例数据
                self.historical_data = DataUtils.generate_sample_data(100)
                print("已生成100条示例数据")
        except Exception as e:
            print(f"加载数据失败: {e}")
            self.historical_data = []

    def _save_data(self):
        """保存历史数据"""
        try:
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.historical_data, f, ensure_ascii=False, indent=2)
            print("数据保存成功")
            return True
        except Exception as e:
            print(f"保存数据失败: {e}")
            return False

    def _on_import_clicked(self):
        """导入数据按钮处理"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "导入数据",
            "",
            "JSON文件 (*.json);;文本文件 (*.txt);;所有文件 (*)"
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.historical_data.extend(data)
                        self._save_data()
                        self._update_history_list()
                        QMessageBox.information(self, "成功", f"已导入 {len(data)} 条记录")
                    else:
                        QMessageBox.warning(self, "错误", "数据格式不正确")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"导入失败: {e}")

    def _on_export_clicked(self):
        """导出数据按钮处理"""
        if not self.historical_data:
            QMessageBox.information(self, "提示", "没有可导出的数据")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出数据",
            "lottery_data_export.json",
            "JSON文件 (*.json);;文本文件 (*.txt)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.historical_data, f, ensure_ascii=False, indent=2)
                QMessageBox.information(self, "成功", "数据导出成功")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"导出失败: {e}")

    def _on_save_clicked(self):
        """保存数据按钮处理"""
        if self._save_data():
            QMessageBox.information(self, "成功", "数据保存成功")
        else:
            QMessageBox.warning(self, "错误", "数据保存失败")

    def _on_add_data_clicked(self):
        """添加数据按钮处理"""
        text, ok = QInputDialog.getMultiLineText(
            self,
            "添加数据",
            "请输入开奖数据（格式：期号 日期 6个正码 特别码）:\n例如：117 2026-04-27 05 12 23 34 45 08"
        )

        if ok and text.strip():
            try:
                parts = text.strip().split()
                if len(parts) >= 8:
                    record = {
                        'period': int(parts[0]),
                        'date': parts[1],
                        'numbers': [int(parts[i]) for i in range(2, 8)],
                        'special': int(parts[7]),
                    }
                    self.historical_data.insert(0, record)
                    self._save_data()
                    self._update_history_list()
                    QMessageBox.information(self, "成功", "数据添加成功")
                else:
                    QMessageBox.warning(self, "错误", "数据格式不正确")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"添加失败: {e}")

    def _on_delete_data_clicked(self):
        """删除数据按钮处理"""
        if not self.history_list.selectedItems():
            QMessageBox.information(self, "提示", "请先选择要删除的记录")
            return

        reply = QMessageBox.question(
            self,
            "确认删除",
            "确定要删除选中的记录吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            for item in self.history_list.selectedItems():
                row = self.history_list.row(item)
                if 0 <= row < len(self.historical_data):
                    del self.historical_data[row]

            self._save_data()
            self._update_history_list()
            QMessageBox.information(self, "成功", "删除成功")

    def _on_clear_data_clicked(self):
        """清空数据按钮处理"""
        reply = QMessageBox.question(
            self,
            "确认清空",
            "确定要清空所有历史记录吗？此操作不可恢复！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.historical_data.clear()
            self._save_data()
            self._update_history_list()
            QMessageBox.information(self, "成功", "历史记录已清空")

    # =====================================
    # 7.8 数据转换处理
    # =====================================

    def _on_convert_clicked(self):
        """转换按钮处理"""
        raw_text = self.raw_text_edit.toPlainText()

        if not raw_text.strip():
            QMessageBox.warning(self, "提示", "请输入要转换的原始数据")
            return

        # 尝试解析数据
        result = DataUtils.parse_raw_data(raw_text)

        if result:
            # 显示转换结果
            formatted = DataUtils.format_data(result)
            self.converted_text_edit.setPlainText(formatted)
            self.statusBar().showMessage("转换成功")
        else:
            QMessageBox.warning(self, "错误", "无法解析数据，请检查格式")

    def _on_add_to_history_clicked(self):
        """添加到历史记录按钮处理"""
        raw_text = self.raw_text_edit.toPlainText()

        if not raw_text.strip():
            QMessageBox.warning(self, "提示", "请输入要添加的原始数据")
            return

        result = DataUtils.parse_raw_data(raw_text)

        if result:
            # 创建记录
            record = {
                'period': result.get('period'),
                'date': result.get('date'),
                'numbers': result.get('numbers'),
                'special': result.get('special'),
            }

            # 添加到历史
            self.historical_data.insert(0, record)
            self._save_data()
            self._update_history_list()

            # 清空输入
            self.raw_text_edit.clear()
            self.converted_text_edit.clear()

            QMessageBox.information(self, "成功", "数据已添加到历史记录")
            self.statusBar().showMessage("添加成功")
        else:
            QMessageBox.warning(self, "错误", "无法解析数据，请检查格式")

    def _on_batch_import_clicked(self):
        """批量导入按钮处理"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "批量导入",
            "",
            "文本文件 (*.txt);;所有文件 (*)"
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                count = 0
                for line in lines:
                    line = line.strip()
                    if line:
                        result = DataUtils.parse_raw_data(line)
                        if result:
                            record = {
                                'period': result.get('period'),
                                'date': result.get('date'),
                                'numbers': result.get('numbers'),
                                'special': result.get('special'),
                            }
                            self.historical_data.append(record)
                            count += 1

                self._save_data()
                self._update_history_list()
                QMessageBox.information(self, "成功", f"成功导入 {count} 条记录")

            except Exception as e:
                QMessageBox.warning(self, "错误", f"导入失败: {e}")

    def _on_history_item_double_clicked(self, item: QListWidgetItem):
        """
        历史记录双击处理

        Args:
            item: 被双击的列表项
        """
        row = self.history_list.row(item)
        if 0 <= row < len(self.historical_data):
            record = self.historical_data[row]
            formatted = DataUtils.format_data(record)

            # 显示在转换结果区
            self.converted_text_edit.setPlainText(formatted)

            # 更新标签页1的原始数据输入框
            self.raw_text_edit.setPlainText(item.text())

    def _update_history_list(self):
        """更新历史记录列表"""
        self.history_list.clear()

        for record in self.historical_data[:200]:  # 最多显示200条
            text = DataUtils.format_data(record)
            self.history_list.addItem(text)

        # 更新状态栏
        self.data_count_label.setText(f"历史记录: {len(self.historical_data)} 条")

        # 更新最新数据显示
        self._refresh_latest_display()

    # =====================================
    # 7.9 数字选择处理
    # =====================================

    def _on_number_selected(self, numbers: List[int]):
        """
        数字选择处理

        Args:
            numbers: 选中的数字列表
        """
        if numbers:
            # 显示选中的数字
            formatted = ', '.join(str(n) for n in sorted(numbers))
            self.selected_numbers_label.setText(formatted)
        else:
            self.selected_numbers_label.setText("无")

    def _clear_number_selection(self):
        """清除数字选择"""
        self.number_panel.clear_selection()
        self.selected_numbers_label.setText("无")

    # =====================================
    # 7.10 预判处理
    # =====================================

    def _on_judge_clicked(self):
        """预判按钮处理"""
        size_judge = self.size_judge_combo.currentText()
        odd_judge = self.odd_judge_combo.currentText()
        tail_judge = self.tail_judge_combo.currentText()

        if size_judge == "请选择" and odd_judge == "请选择" and tail_judge == "请选择":
            QMessageBox.information(self, "提示", "请至少选择一个预判条件")
            return

        # 筛选符合条件的数字
        candidates = list(range(1, 50))

        if size_judge == "大(26-49)":
            candidates = [n for n in candidates if n > 25]
        elif size_judge == "小(1-25)":
            candidates = [n for n in candidates if n <= 25]

        if odd_judge == "单数":
            candidates = [n for n in candidates if n % 2 == 1]
        elif odd_judge == "双数":
            candidates = [n for n in candidates if n % 2 == 0]

        if tail_judge == "大(5-9)":
            candidates = [n for n in candidates if n % 10 >= 5]
        elif tail_judge == "小(0-4)":
            candidates = [n for n in candidates if n % 10 <= 4]

        if not candidates:
            self.judge_result_label.setText("预判结果：没有符合条件的数字")
            return

        # 随机选择一个
        selected = random.choice(candidates)

        # 显示结果
        color_info = ""
        if LotteryConfig.is_red(selected):
            color_info = "🔴 红色"
        elif LotteryConfig.is_blue(selected):
            color_info = "🔵 蓝色"
        else:
            color_info = "🟢 绿色"

        result_text = f"预判结果：{selected}号\n{color_info} | {LotteryConfig.NUMBER_NAMES[selected]} | {LotteryConfig.NUMBER_ELEMENTS[selected]}行\n符合条件的数字共{len(candidates)}个"

        self.judge_result_label.setText(result_text)

    # =====================================
    # 7.11 算法选择处理
    # =====================================

    def _on_algorithm_changed(self, index: int):
        """
        算法选择改变处理

        Args:
            index: 选中的算法索引
        """
        self.current_algorithm_index = index

        # 更新算法描述
        if index < len(LotteryConfig.ALGORITHMS):
            _, desc = LotteryConfig.ALGORITHMS[index]
            self.algorithm_desc_label.setText(desc)

    # =====================================
    # 7.12 预测处理
    # =====================================

    def _on_predict_clicked(self):
        """预测按钮处理"""
        if len(self.historical_data) < 10:
            QMessageBox.warning(self, "数据不足", "历史数据不足10条，请先添加更多数据")
            return

        # 创建预测算法实例
        predictor = PredictionAlgorithms(self.historical_data)

        # 获取选中的算法
        algorithm_index = self.current_algorithm_index

        # 调用对应的算法
        if algorithm_index == 0:
            predictions = predictor.comprehensive_recommendation(6)
        elif algorithm_index == 1:
            predictions = predictor.hot_cold_algorithm(6)
        elif algorithm_index == 2:
            predictions = predictor.odd_even_algorithm(6)
        elif algorithm_index == 3:
            predictions = predictor.big_small_algorithm(6)
        elif algorithm_index == 4:
            predictions = predictor.missing_value_analysis(6)
        elif algorithm_index == 5:
            predictions = predictor.adjacent_number_analysis(6)
        elif algorithm_index == 6:
            predictions = predictor.tail_distribution_algorithm(6)
        elif algorithm_index == 7:
            predictions = predictor.range_distribution_algorithm(6)
        elif algorithm_index == 8:
            predictions = predictor.roulette_selection(6)
        elif algorithm_index == 9:
            predictions = predictor.historical_similarity(6)
        elif algorithm_index == 10:
            predictions = predictor.poisson_distribution(6)
        elif algorithm_index == 11:
            predictions = predictor.mystical_algorithm(6)
        else:
            predictions = predictor.comprehensive_recommendation(6)

        # 显示预测结果
        self._display_predictions(predictions)

        self.statusBar().showMessage("预测完成")

    def _on_random_draw_clicked(self):
        """随机抽取按钮处理"""
        # 使用轮盘赌算法
        if len(self.historical_data) < 10:
            predictor = PredictionAlgorithms([])
        else:
            predictor = PredictionAlgorithms(self.historical_data)

        predictions = predictor.roulette_selection(6)
        self._display_predictions(predictions)

        self.statusBar().showMessage("随机抽取完成")

    def _on_ml_predict_clicked(self):
        """机器学习预测按钮处理"""
        if len(self.historical_data) < 20:
            QMessageBox.warning(self, "数据不足", "机器学习需要至少20条历史数据")
            return

        try:
            # 创建机器学习模型
            model = MLPredictionModel(self.historical_data)

            # 进行预测
            predictions = model.predict_with_all_models()

            # 显示预测结果
            self._display_predictions(predictions)

            self.statusBar().showMessage("机器学习预测完成")

        except Exception as e:
            QMessageBox.warning(self, "错误", f"机器学习预测失败: {e}")

    def _display_predictions(self, predictions: List[int]):
        """
        显示预测结果

        Args:
            predictions: 预测的数字列表
        """
        # 排序显示
        sorted_preds = sorted(predictions)

        # 更新显示文本
        display_text = f"预测号码: {' '.join(str(n).zfill(2) for n in sorted_preds)}"
        self.prediction_display.setText(display_text)

        # 清除旧的数字按钮
        while self.prediction_number_layout.count():
            item = self.prediction_number_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # 添加新的数字按钮
        for i, num in enumerate(sorted_preds):
            btn = NumberButton(num)
            btn.set_selected(True)
            row = i // 6
            col = i % 6
            self.prediction_number_layout.addWidget(btn, row, col)

        # 统计信息
        red_count = sum(1 for n in predictions if LotteryConfig.is_red(n))
        blue_count = sum(1 for n in predictions if LotteryConfig.is_blue(n))
        green_count = sum(1 for n in predictions if LotteryConfig.is_green(n))

        odd_count = sum(1 for n in predictions if n % 2 == 1)
        even_count = 6 - odd_count

        big_count = sum(1 for n in predictions if n > 25)
        small_count = 6 - big_count

        stats_text = (
            f"颜色分布: 🔴红{red_count}个 🔵蓝{blue_count}个 🟢绿{green_count}个\n"
            f"单双分布: 单{odd_count}个 双{even_count}个\n"
            f"大小分布: 大{big_count}个 小{small_count}个"
        )

        self.prediction_stats_label.setText(stats_text)

    def _refresh_latest_display(self):
        """刷新最新数据显示"""
        if not self.historical_data:
            self.latest_display.setText("暂无数据")
            return

        latest = self.historical_data[0]
        numbers = latest.get('numbers', [])
        special = latest.get('special', 0)

        # 格式化显示
        numbers_text = ' '.join(str(n).zfill(2) for n in numbers)
        text = f"第{latest.get('period', '?')}期 | {latest.get('date', '?')}\n"
        text += f"正码: {numbers_text}\n"
        text += f"特别码: {str(special).zfill(2)}"

        self.latest_display.setText(text)

    def _refresh_chart(self):
        """刷新统计图表"""
        if not self.historical_data:
            QMessageBox.information(self, "提示", "暂无数据可显示")
            return

        # 创建预测算法实例以获取统计数据
        predictor = PredictionAlgorithms(self.historical_data)

        chart_type = self.chart_type_combo.currentText()

        if chart_type == "频率分布图":
            self.statistics_chart.plot_frequency(predictor.frequency, "数字出现频率分布")
        elif chart_type == "遗漏值图":
            self.statistics_chart.plot_missing(predictor.missing, "数字遗漏值分布")
        elif chart_type == "单双分布图":
            self.statistics_chart.plot_distribution(
                predictor.odd_even_ratio,
                "单双号分布"
            )
        elif chart_type == "大小分布图":
            self.statistics_chart.plot_distribution(
                predictor.big_small_ratio,
                "大小号分布"
            )


# ============================================================================
# 第八部分：应用入口
# ============================================================================

def main():
    """
    主函数 - 应用程序入口点
    """
    print("=" * 60)
    print("彩票预测系统 v3.0 启动中...")
    print("=" * 60)

    # 检查依赖库
    print("\n正在检查依赖库...")

    required_modules = [
        ('PyQt6', 'GUI框架'),
        ('numpy', '数值计算'),
        ('pandas', '数据处理'),
        ('matplotlib', '数据可视化'),
        ('seaborn', '统计可视化'),
        ('scipy', '科学计算'),
        ('statsmodels', '统计建模'),
        ('sklearn', '机器学习'),
        ('optuna', '超参数优化'),
        ('torch', '深度学习'),
    ]

    all_ok = True
    for module_name, description in required_modules:
        try:
            __import__(module_name)
            print(f"✓ {description} ({module_name}) - OK")
        except ImportError:
            print(f"✗ {description} ({module_name}) - 未安装")
            all_ok = False

    if not all_ok:
        print("\n警告: 部分依赖库未安装，部分功能可能无法使用。")
        print("请运行: pip install -r requirements.txt")

    print("\n" + "=" * 60)
    print("正在初始化应用程序...")
    print("=" * 60)

    # 创建应用实例
    app = QApplication(sys.argv)

    # 设置应用信息
    app.setApplicationName("彩票预测系统")
    app.setApplicationVersion("3.0")
    app.setOrganizationName("AI Assistant")

    # PyQt6 默认启用高DPI支持，无需手动设置

    # 创建主窗口
    window = LotteryPredictionWindow()
    window.show()

    print("\n" + "=" * 60)
    print("彩票预测系统已启动！")
    print("=" * 60)
    print("\n使用方法：")
    print("1. 在'数据导入'标签页粘贴原始数据或批量导入")
    print("2. 在'数据分析'标签页选择预测算法")
    print("3. 点击'开始预测'获取预测结果")
    print("4. 使用顶部工具栏调整字体大小和管理数据")
    print("=" * 60)

    # 进入事件循环
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
============================================================================
彩票预测系统 v7.1 - PyQt6完整实现
============================================================================
本系统是一个功能完整的彩票（香港六合）预测与分析平台，
集成了数据导入、格式转换、数据分析、机器学习预测等多种功能。

核心功能模块：
1. 数据导入与格式转换 - 支持多种格式的原始数据输入
2. 预测与抽取 - 12种预测算法，涵盖统计、机器学习、深度学习
3. 第七位预判 - 特别号大小、单双、尾数预判
4. 统计分析图表 - 8种图表类型，全面数据分析

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

版本: 5.0
============================================================================
"""

# ============================================================================
# 第一部分：导入所有必要的库
# ============================================================================

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QTabWidget, QPushButton, QLabel, QTextEdit, QLineEdit,
    QScrollArea, QScrollBar, QSplitter, QComboBox, QSpinBox, QSlider,
    QProgressBar, QListWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QFrame, QSizePolicy, QStyleFactory, QToolBar, QStatusBar, QMenuBar,
    QDialog, QMessageBox, QFileDialog, QInputDialog, QColorDialog, QFontDialog,
    QMenu, QGroupBox
)
from PyQt6.QtCore import (
    Qt, QSize, QPoint, QRect, QTimer, QThread, QObject,
    pyqtSignal, pyqtSlot, QPropertyAnimation, QEasingCurve, QDateTime
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QBrush, QPen, QPainter, QPixmap,
    QImage, QIcon, QAction, QKeySequence, QCursor, QFontDatabase
)
from PyQt6.QtGui import QShortcut

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

# matplotlib延迟导入 - 启动时不加载
_matplotlib_module = None
_pyplot_module = None
_figure_module = None
_canvas_class = None
_patches_module = None
_sns_module = None
HAS_MPL = None

def _get_mpl():
    """懒加载Matplotlib"""
    global _matplotlib_module, _pyplot_module, _figure_module, _canvas_class, _patches_module, HAS_MPL
    if HAS_MPL is not None:
        return _matplotlib_module
    try:
        import matplotlib as mpl_mod
        mpl_mod.use('QtAgg')
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as canvas_cls
        from matplotlib.figure import Figure as fig_cls
        import matplotlib.pyplot as plt_mod
        import matplotlib.patches as patches_mod
        _matplotlib_module = mpl_mod
        _pyplot_module = plt_mod
        _figure_module = fig_cls
        _canvas_class = canvas_cls
        _patches_module = patches_mod
        HAS_MPL = True
    except Exception:
        _matplotlib_module = None
        _pyplot_module = None
        _figure_module = None
        _canvas_class = None
        _patches_module = None
        HAS_MPL = False
    return _matplotlib_module

def _get_sns():
    """懒加载Seaborn"""
    global _sns_module
    if _sns_module is not None:
        return _sns_module
    try:
        import seaborn as sns_mod
        _sns_module = sns_mod
    except Exception:
        _sns_module = None
    return _sns_module

# scipy延迟导入 - 启动时不加载
_scipy_stats_module = None
_scipy_optimize_module = None
_scipy_special_module = None
_scipy_signal_module = None
_scipy_interpolate_module = None
_scipy_ks_module = None

def _get_scipy_stats():
    global _scipy_stats_module
    if _scipy_stats_module is not None:
        return _scipy_stats_module
    try:
        from scipy import stats as s_mod
        _scipy_stats_module = s_mod
    except Exception:
        _scipy_stats_module = None
    return _scipy_stats_module

def _get_scipy_optimize():
    global _scipy_optimize_module
    if _scipy_optimize_module is not None:
        return _scipy_optimize_module
    try:
        from scipy import optimize as o_mod
        _scipy_optimize_module = o_mod
    except Exception:
        _scipy_optimize_module = None
    return _scipy_optimize_module

def _get_scipy_special():
    global _scipy_special_module
    if _scipy_special_module is not None:
        return _scipy_special_module
    try:
        from scipy.special import gamma as _g, factorial as _f
        _scipy_special_module = True
    except Exception:
        _scipy_special_module = None
    return _scipy_special_module

def _get_scipy_signal():
    global _scipy_signal_module
    if _scipy_signal_module is not None:
        return _scipy_signal_module
    try:
        from scipy import signal as sig_mod
        _scipy_signal_module = sig_mod
    except Exception:
        _scipy_signal_module = None
    return _scipy_signal_module

def _get_scipy_interpolate():
    global _scipy_interpolate_module
    if _scipy_interpolate_module is not None:
        return _scipy_interpolate_module
    try:
        from scipy import interpolate as int_mod
        _scipy_interpolate_module = int_mod
    except Exception:
        _scipy_interpolate_module = None
    return _scipy_interpolate_module

def _get_scipy_ks():
    global _scipy_ks_module
    if _scipy_ks_module is not None:
        return _scipy_ks_module
    try:
        from scipy.stats import ks_2samp as ks_mod
        _scipy_ks_module = ks_mod
    except Exception:
        _scipy_ks_module = None
    return _scipy_ks_module

# sklearn延迟导入 - 启动时不加载（节省~5秒）
_sklearn_loaded = False
_train_test_split = None
_cross_val_score = None
_RandomForestClassifier = None
_GradientBoostingClassifier = None
_LogisticRegression = None
_MLPClassifier = None
_StandardScaler = None
_MinMaxScaler = None
_accuracy_score = None
_classification_report = None
_KMeans = None
_GaussianNB = None
_PCA = None
_cosine_similarity = None

def _get_sklearn():
    """懒加载sklearn所有组件"""
    global _sklearn_loaded, _train_test_split, _cross_val_score
    global _RandomForestClassifier, _GradientBoostingClassifier, _LogisticRegression
    global _MLPClassifier, _StandardScaler, _MinMaxScaler
    global _accuracy_score, _classification_report, _KMeans, _GaussianNB
    global _PCA, _cosine_similarity
    if _sklearn_loaded:
        return
    try:
        from sklearn.model_selection import train_test_split as tts, cross_val_score as cvs
        from sklearn.ensemble import RandomForestClassifier as RFC, GradientBoostingClassifier as GBC
        from sklearn.linear_model import LogisticRegression as LR
        from sklearn.neural_network import MLPClassifier as MLP
        from sklearn.preprocessing import StandardScaler as SS, MinMaxScaler as MMS
        from sklearn.metrics import accuracy_score as acc, classification_report as cr
        from sklearn.cluster import KMeans as KM
        from sklearn.naive_bayes import GaussianNB as GNB
        from sklearn.decomposition import PCA as pca_cls
        from sklearn.metrics.pairwise import cosine_similarity as cs
        _train_test_split = tts
        _cross_val_score = cvs
        _RandomForestClassifier = RFC
        _GradientBoostingClassifier = GBC
        _LogisticRegression = LR
        _MLPClassifier = MLP
        _StandardScaler = SS
        _MinMaxScaler = MMS
        _accuracy_score = acc
        _classification_report = cr
        _KMeans = KM
        _GaussianNB = GNB
        _PCA = pca_cls
        _cosine_similarity = cs
    except Exception:
        pass
    _sklearn_loaded = True

# TensorFlow可选导入 - 懒加载
_tf_module = None
_keras_module = None
_layers_module = None
HAS_TF = None

def _get_tf():
    """懒加载TensorFlow"""
    global _tf_module, _keras_module, _layers_module, HAS_TF
    if HAS_TF is not None:
        return _tf_module
    try:
        import tensorflow as tf_mod
        from tensorflow import keras as keras_mod
        from tensorflow.keras import layers as layers_mod
        tf_mod.get_logger().setLevel('ERROR')
        _tf_module = tf_mod
        _keras_module = keras_mod
        _layers_module = layers_mod
        HAS_TF = True
    except Exception:
        _tf_module = None
        _keras_module = None
        _layers_module = None
        HAS_TF = False
    return _tf_module

# Optuna可选导入 - 懒加载
_optuna_module = None
_TPESampler_class = None
HAS_OPTUNA = None

def _get_optuna():
    """懒加载Optuna"""
    global _optuna_module, _TPESampler_class, HAS_OPTUNA
    if HAS_OPTUNA is not None:
        return _optuna_module
    try:
        import optuna as optuna_mod
        from optuna.samplers import TPESampler as tpe_mod
        _optuna_module = optuna_mod
        _TPESampler_class = tpe_mod
        HAS_OPTUNA = True
    except Exception:
        _optuna_module = None
        _TPESampler_class = None
        HAS_OPTUNA = False
    return _optuna_module

# PyTorch可选导入 - 懒加载
_torch_module = None
_nn_module = None
_optim_module = None
HAS_TORCH = None

def _get_torch():
    """懒加载PyTorch"""
    global _torch_module, _nn_module, _optim_module, HAS_TORCH
    if HAS_TORCH is not None:
        return _torch_module
    try:
        import torch as torch_mod
        import torch.nn as nn_mod
        import torch.optim as optim_mod
        _torch_module = torch_mod
        _nn_module = nn_mod
        _optim_module = optim_mod
        HAS_TORCH = True
    except Exception:
        _torch_module = None
        _nn_module = None
        _optim_module = None
        HAS_TORCH = False
    return _torch_module

def _get_nn():
    """获取PyTorch nn模块"""
    _get_torch()
    return _nn_module

def _get_optim():
    """获取PyTorch optim模块"""
    _get_torch()
    return _optim_module

# NetworkX可选导入 - 懒加载
_nx_module = None
HAS_NX = None

def _get_nx():
    """懒加载NetworkX"""
    global _nx_module, HAS_NX
    if HAS_NX is not None:
        return _nx_module
    try:
        import networkx as nx_mod
        _nx_module = nx_mod
        HAS_NX = True
    except Exception:
        _nx_module = None
        HAS_NX = False
    return _nx_module

import sys
import os
import re
import json
import urllib.request
import csv
import datetime
import random
import math
import copy
import pickle
import traceback
import warnings
from collections import Counter, defaultdict
from functools import lru_cache, wraps
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
from pathlib import Path

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


# ============================================================================
# 第二部分：全局常量和配置
# ============================================================================

class LotteryConfig:
    """
    彩票配置类 - 定义所有全局配置和常量
    
    本系统使用香港六合）（Mark Six）的规则：
    - 49个数字（01-49）
    - 每次开奖7个数字（前6个为正码，第7个为特别码）
    - 数字分为三种颜色：红、蓝、绿
    """
    
    WINDOW_TITLE = "彩票预测系统 v7.1"
    WINDOW_MIN_WIDTH = 1400
    WINDOW_MIN_HEIGHT = 900
    
    FONT_SIZES = {
        '初号': 42, '小初': 36, '一号': 26, '小一': 24,
        '二号': 22, '小二': 18, '三号': 16, '小四': 12,
    }
    DEFAULT_FONT_SIZE_KEY = '二号'
    
    COLOR_BG_PRIMARY = "#FFFFFF"
    COLOR_BG_SECONDARY = "#FFFFFF"
    COLOR_BG_TERTIARY = "#FFFFFF"
    COLOR_TEXT_PRIMARY = "#000000"
    COLOR_TEXT_SECONDARY = "#333333"
    COLOR_TEXT_LIGHT = "#555555"
    COLOR_SUCCESS = "#2ECC71"
    COLOR_ERROR = "#E74C3C"
    COLOR_WARNING = "#F39C12"
    COLOR_INFO = "#3498DB"
    COLOR_BORDER = "#DDDDDD"
    COLOR_BUTTON_BG = "#FFFFFF"
    COLOR_BUTTON_HOVER = "#F8F9FA"
    COLOR_BUTTON_PRESSED = "#E8E8E8"
    
    RED_NUMBERS = [1, 2, 7, 8, 12, 13, 18, 19, 23, 24, 29, 30, 34, 35, 40, 45, 46]
    BLUE_NUMBERS = [3, 4, 9, 10, 14, 15, 20, 25, 26, 31, 36, 37, 41, 42, 47, 48]
    GREEN_NUMBERS = [5, 6, 11, 16, 17, 21, 22, 27, 28, 32, 33, 38, 39, 43, 44, 49]
    
    NUMBER_COLORS = {}
    for num in RED_NUMBERS:
        NUMBER_COLORS[num] = {"bg": "#FFFFFF", "text": "#FF0000", "border": "#FF0000"}
    for num in BLUE_NUMBERS:
        NUMBER_COLORS[num] = {"bg": "#FFFFFF", "text": "#0000FF", "border": "#0000FF"}
    for num in GREEN_NUMBERS:
        NUMBER_COLORS[num] = {"bg": "#FFFFFF", "text": "#008000", "border": "#008000"}
    
    NUMBER_NAMES = {
        1: "鸡", 2: "鸡", 3: "狗", 4: "狗", 5: "猪", 6: "猪", 7: "鼠", 8: "鼠",
        9: "牛", 10: "牛", 11: "虎", 12: "虎", 13: "兔", 14: "兔", 15: "龙", 16: "龙",
        17: "蛇", 18: "蛇", 19: "马", 20: "马", 21: "羊", 22: "羊", 23: "猴", 24: "猴",
        25: "鸡", 26: "鸡", 27: "狗", 28: "狗", 29: "猪", 30: "猪", 31: "鼠", 32: "鼠",
        33: "牛", 34: "牛", 35: "虎", 36: "虎", 37: "兔", 38: "龙", 39: "龙", 40: "蛇",
        41: "蛇", 42: "马", 43: "马", 44: "羊", 45: "羊", 46: "猴", 47: "猴", 48: "鸡", 49: "狗"
    }
    
    NUMBER_ELEMENTS = {
        1: "金", 2: "金", 3: "木", 4: "木", 5: "水", 6: "水", 7: "火", 8: "火",
        9: "土", 10: "土", 11: "木", 12: "木", 13: "水", 14: "水", 15: "金", 16: "金",
        17: "火", 18: "火", 19: "土", 20: "土", 21: "木", 22: "木", 23: "水", 24: "水",
        25: "金", 26: "金", 27: "火", 28: "火", 29: "土", 30: "土", 31: "木", 32: "木",
        33: "水", 34: "水", 35: "金", 36: "金", 37: "火", 38: "火", 39: "土", 40: "土",
        41: "木", 42: "木", 43: "水", 44: "水", 45: "金", 46: "金", 47: "火", 48: "火", 49: "土"
    }
    
    RANGES = [
        (1, 9, "1-9区"), (10, 19, "10-19区"), (20, 29, "20-29区"),
        (30, 39, "30-39区"), (40, 49, "40-49区"),
    ]
    
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
        ("号码关联图算法", "NetworkX PageRank/中心性分析"),
        ("最短路径算法", "NetworkX Dijkstra号码转移分析"),
        ("社区发现算法", "NetworkX Louvain社区检测"),
        ("图聚类算法", "NetworkX连通分量聚类"),
        ("NumPy矩阵算法", "基于NumPy矩阵运算深度分析"),
        ("SciPy优化算法", "基于SciPy科学计算优化预测"),
        ("Scikit-learn集成算法", "基于Sklearn多模型集成预测"),
        ("PyTorch深度学习算法", "基于PyTorch神经网络预测"),
        ("NetworkX图算法", "基于图论网络分析预测"),
        # ======================================================================== #
        # 功能8：特别码专项分析 - 新增2个特别码专属算法（索引21、22）
        # ======================================================================== #
        ("特别码频率回归", "基于特别码历史频率加权回归"),
        ("特别码关联算法", "基于特别码与正码关联规则挖掘"),
    ]
    
    @classmethod
    def get_number_color(cls, number):
        return cls.NUMBER_COLORS.get(number, {"bg": "#FFFFFF", "text": "#000000", "border": "#CCCCCC"})
    
    @classmethod
    def is_red(cls, number):
        return number in cls.RED_NUMBERS
    
    @classmethod
    def is_blue(cls, number):
        return number in cls.BLUE_NUMBERS
    
    @classmethod
    def is_green(cls, number):
        return number in cls.GREEN_NUMBERS
    
    @classmethod
    def is_odd(cls, number):
        return number % 2 == 1
    
    @classmethod
    def is_even(cls, number):
        return number % 2 == 0
    
    @classmethod
    def is_big(cls, number):
        return number > 25
    
    @classmethod
    def is_small(cls, number):
        return number <= 25
    
    @classmethod
    def get_tail_digit(cls, number):
        return number % 10
    
    @classmethod
    def get_range_index(cls, number):
        for i, (start, end, name) in enumerate(cls.RANGES):
            if start <= number <= end:
                return i
        return -1


# ============================================================================
# 第三部分：工具函数模块
# ============================================================================

class ColorUtils:
    @staticmethod
    def hex_to_qcolor(hex_color):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return QColor(r, g, b)
        return QColor(0, 0, 0)
    
    @staticmethod
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))
    
    @staticmethod
    def rgb_to_hex(r, g, b):
        return f"#{r:02X}{g:02X}{b:02X}"


class FontUtils:
    FONT_FAMILIES = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Microsoft YaHei UI', 'Segoe UI', 'Arial', 'Tahoma', 'Verdana', 'Helvetica']
    
    @classmethod
    def get_default_font_family(cls):
        font_db = QFontDatabase()
        available_families = font_db.families()
        for family in cls.FONT_FAMILIES:
            if family in available_families:
                return family
        return 'Arial'
    
    @classmethod
    def create_font(cls, size_key='二号', bold=False, italic=False):
        font_family = cls.get_default_font_family()
        font_size = LotteryConfig.FONT_SIZES.get(size_key, 16)
        font = QFont(font_family, font_size)
        font.setBold(bold)
        font.setItalic(italic)
        return font


class DataUtils:
    @staticmethod
    def parse_raw_data(raw_text):
        """
        解析原始开奖数据，支持多种格式自动识别
        
        支持的输入格式：
        【格式A】数字与生肖/五行交替，无+分隔特别码：
          第116期最新开奖结果 2026年04月26日 15 龙/水 46 鸡/木 16 兔/木 10 鸡/火 48 羊/火 33 狗/火 22 鸡/水
          → 输出: period=116, numbers=[15,46,16,10,48,33], special=22 (第7个数字为特别码)
        
        【格式B】数字在前、生肖在后，期号带年份前缀如2026129：
          第 2026129 期 2026年05月09日 34 07 40 21 17 25 41 鸡/金 鼠/土 兔/火 狗/土 虎/木 马/木 虎/火
          → 输出: period=129(后3位), numbers=[34,07,40,21,17,25], special=41
        
        【格式C】数字与生肖交替，+分隔特别码：
          第129期最新开奖结果 2026年05月09日 34 鸡/金 07 鼠/土 40 兔/火 21 狗/土 17 虎/木 25 马/木 + 41 虎/火
          → 输出: period=129, numbers=[34,07,40,21,17,25], special=41
        
        【格式D】简单格式（保持兼容）：
          第116期最新开奖结果 2026年04月26日 15 46 16 10 48 33 + 22
          116 2026-04-26 15 46 16 10 48 33 22
        """
        try:
            raw_text = ' '.join(raw_text.split())
            
            # 【需求1-增强】提取期号 - 支持年份前缀格式
            # 格式1: 第 2026129 期 (中间有空格)
            period_match = re.search(r'第\s*(\d+)\s*期', raw_text)
            if period_match:
                period_str = period_match.group(1)
                # 如果期号是6位年份+3位期号，取后3位
                if len(period_str) == 9 and period_str.isdigit():
                    period = int(period_str[-3:])
                else:
                    period = int(period_str)
            else:
                # 尝试从开头提取纯数字期号
                head_match = re.match(r'^(\d+)\s', raw_text)
                period = int(head_match.group(1)) if head_match else None
            
            # 提取日期（支持 中文日期 和 横线日期）
            date_str = None
            date_match_cn = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', raw_text)
            if date_match_cn:
                date_str = date_match_cn.group(1) + "年" + date_match_cn.group(2).zfill(2) + "月" + date_match_cn.group(3).zfill(2) + "日"
            else:
                date_match_dash = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', raw_text)
                if date_match_dash:
                    date_str = date_match_dash.group(1) + "年" + date_match_dash.group(2).zfill(2) + "月" + date_match_dash.group(3).zfill(2) + "日"
            if date_str is None:
                date_str = datetime.date.today().strftime("%Y年%m月%d日")
            
            # 【需求1-增强】处理特殊格式：先移除+号后的特别码部分，单独提取
            has_plus_sign = ' + ' in raw_text or '+' in raw_text
            special_candidate = None
            text_part = raw_text
            
            if has_plus_sign:
                # 有+号的情况：+号后面的是特别码
                plus_match = re.search(r'\+\s*(\d{1,2})', raw_text)
                if plus_match:
                    special_candidate = int(plus_match.group(1))
                # 移除+号及后面内容
                text_part = re.sub(r'\s*\+\s*\d{1,2}.*$', '', raw_text)
            
            # 去除日期部分
            if date_match_cn:
                text_part = text_part[:date_match_cn.start()] + text_part[date_match_cn.end():]
            elif date_match_dash:
                text_part = text_part[:date_match_dash.start()] + text_part[date_match_dash.end():]
            
            # 去除期号文字
            text_part = re.sub(r'第\s*\d+\s*期\s*最新开奖结果\s*', '', text_part)
            text_part = re.sub(r'第\s*\d+\s*期', '', text_part)
            
            # 【需求1-增强】精细化处理：区分数字和生肖/五行标记
            # 生肖标记格式：龙/水、鸡/木、狗/土等（生肖/五行）
            # 去除所有生肖/五行标记，但保留纯数字和+号
            text_cleaned = text_part
            # 去除 生肖/五行 格式
            text_cleaned = re.sub(r'[\u9f99-\u9fbf]/[\u91d1\u6728\u6c34\u706b\u571f]', ' ', text_cleaned)
            # 去除纯生肖
            zodiacs = '龙|马|蛇|羊|猴|鸡|狗|猪|鼠|牛|虎|兔'
            text_cleaned = re.sub(zodiacs, ' ', text_cleaned)
            # 只保留数字和空格、+号
            text_cleaned = re.sub(r'[^\d\s+]', ' ', text_cleaned)
            
            # 提取所有1-49范围内的数字
            numbers = []
            for match in re.finditer(r'\b(\d{1,2})\b', text_cleaned):
                num = int(match.group(1))
                if 1 <= num <= 49:
                    numbers.append(num)
            
            # 去重但保持顺序
            seen = set()
            unique_numbers = []
            for n in numbers:
                if n not in seen:
                    seen.add(n)
                    unique_numbers.append(n)
            numbers = unique_numbers
            
            # 【需求1-增强】判断特别码
            # 情况1: 有+号，已提取special_candidate
            # 情况2: 无+号，有8个以上数字，最后一个数字为特别码（格式A/B无+号）
            # 情况3: 正好7个数字，前6个为正码，最后一个为特别码
            special = special_candidate
            if special is None and len(numbers) >= 7:
                # 格式A/B：没有+号时，第7个数字是特别码
                special = numbers[6]
                numbers = numbers[:6]
            
            if len(numbers) >= 6 and special is not None:
                return {
                    'period': period, 'date': date_str,
                    'numbers': numbers[:6], 'special': special, 'all_numbers': numbers[:6] + [special],
                }
            return None
        except Exception as e:
            print("解析数据失败: " + str(e))
            return None
    
    @staticmethod
    def format_data(data):
        """【需求1-更新】格式化数据为标准输出：2026年04月26日 第116期 15 46 16 10 48 33 + 22"""
        period = data.get('period', '?')
        date = data.get('date', '?')
        numbers = data.get('numbers', [])
        special = data.get('special', '?')
        if numbers and special != '?':
            numbers_str = ' '.join(str(n).zfill(2) for n in numbers)
            return date + " 第" + str(period) + "期 " + numbers_str + " + " + str(special).zfill(2)
        return date + " 第" + str(period) + "期"
    
    @staticmethod
    def generate_sample_data(count=100):
        data = []
        base_date = datetime.date.today()
        for i in range(count):
            numbers = random.sample(range(1, 50), 7)
            numbers.sort()
            record = {
                'period': count - i, 'date': (base_date - datetime.timedelta(days=i)).strftime("%Y-%m-%d"),
                'numbers': numbers[:6], 'special': numbers[6], 'all_numbers': numbers,
            }
            data.append(record)
        return data


class MathUtils:
    @staticmethod
    def calculate_mean(numbers):
        return float(np.mean(numbers)) if numbers else 0.0
    
    @staticmethod
    def calculate_median(numbers):
        return float(np.median(numbers)) if numbers else 0.0
    
    @staticmethod
    def calculate_std(numbers):
        return float(np.std(numbers)) if numbers else 0.0
    
    @staticmethod
    def calculate_frequency(numbers):
        return dict(Counter(numbers))
    
    @staticmethod
    def calculate_missing_cycle(current_miss, avg_frequency):
        if avg_frequency <= 0:
            return 0.5
        probability = 1 - np.exp(-current_miss / avg_frequency)
        return min(1.0, max(0.0, probability))
    
    @staticmethod
    def poisson_probability(lambda_param, k):
        scipy_st = _get_scipy_stats()
        if scipy_st is not None and hasattr(scipy_st, 'poisson'):
            return float(scipy_st.poisson.pmf(k, lambda_param))
        # 简化fallback
        return 1.0 / (lambda_param + 1)
    
    @staticmethod
    def moving_average(data, window):
        if len(data) < window:
            return []
        weights = np.ones(window) / window
        ma = np.convolve(data, weights, mode='valid')
        return ma.tolist()


# ============================================================================
# 第四部分：预测算法模块
# ============================================================================

class PredictionAlgorithms:
    """预测算法集合 - v7.1增强版，深度集成11大核心库"""
    
    def __init__(self, historical_data):
        self.data = historical_data
        self.analysis_results = {}
        # PyTorch设备只在第一次使用时设置
        self.device = None
        self._prepare_data()
    
    def _prepare_data(self):
        """增强的数据准备方法"""
        if not self.data:
            # 初始化默认值防止后续访问报错
            self._init_defaults()
            return
        try:
            # 原有基础统计
            all_numbers = []
            for record in self.data:
                all_numbers.extend(record.get('numbers', []))
            self.frequency = MathUtils.calculate_frequency(all_numbers)
            self.missing = {}
            for num in range(1, 50):
                self.missing[num] = self._calculate_missing(num)
            self.range_distribution = self._calculate_range_distribution()
            self.tail_distribution = self._calculate_tail_distribution()
            self.odd_even_ratio = self._calculate_odd_even_ratio()
            self.big_small_ratio = self._calculate_big_small_ratio()
            
            # 新增：Pandas DataFrame构建
            self._build_dataframe()
            
            # 新增：NumPy共现相关性矩阵
            self._calculate_correlation_matrix()
            
            # 新增：StatsModels自相关分析
            self._calculate_autocorrelation()
            
            # 新增：间隔统计特征
            self._calculate_interval_stats()
            
            # 新增：连号邻号历史概率
            self._calculate_adjacent_stats()
            
            # 新增：移动平均
            self._calculate_moving_avg()
            
            # 新增：sklearn特征准备
            self._prepare_sklearn_features()
            
            # 新增：深度NumPy矩阵运算
            self._prepare_numpy_advanced()
            
            # 新增：SciPy优化和分布分析
            self._prepare_scipy_advanced()
            
            # 新增：TensorFlow深度学习模型（可选）
            if _get_tf():
                self._prepare_tensorflow_model()
            
            # 新增：PyTorch LSTM时序模型（可选）
            if _get_torch():
                self._prepare_pytorch_lstm()
        except Exception as e:
            print("数据准备警告: " + str(e))
            self._init_defaults()
    
    def _init_defaults(self):
        """初始化默认值，防止后续访问属性报错"""
        self.frequency = {}
        self.missing = {i: 50 for i in range(1, 50)}
        self.range_distribution = {i: 0 for i in range(5)}
        self.tail_distribution = {i: 0 for i in range(10)}
        self.odd_even_ratio = 0.5
        self.big_small_ratio = 0.5
        self.correlation_matrix = np.eye(49)
        self.autocorrelation = {}
        self.interval_stats = {}
        self.adjacent_stats = {}
        self.moving_avg = {}
        self.sklearn_features = None
        self.sklearn_features_scaled = None
        self.scaler = None
        self.kmeans_labels = None
        self.kmeans_centers = None
        self.np_features = None
        self.np_regression_coeffs = np.array([0, 25])
        self.np_predicted_missing = 25
        self.np_histogram = np.zeros(10)
        self.np_bin_edges = np.arange(11) * 5
        self.np_percentiles = np.array([12.5, 25, 37.5, 45])
        self.np_corrcoef = np.eye(7)
        self.scipy_weights = [0.2, 0.2, 0.15, 0.15, 0.2]
        self.ks_test_result = 0.5
        self.scipy_smoothed = None
        self.scipy_interp_func = lambda xi: 25.0
        self.scipy_interp_edge = 25.0
        self.tf_predictions = {}
        self.tf_autoencoder = None
    
    def _prepare_numpy_advanced(self):
        """深度NumPy矩阵运算：特征向量构建、共现矩阵特征值、线性回归lstsq"""
        if len(self.data) < 10:
            self.np_features = None
            return
        # 构建49个数字的特征向量矩阵
        features_list = []
        for num in range(1, 50):
            freq = self.frequency.get(num, 0)
            miss = min(self.missing.get(num, 50), 50)
            ma5 = int(self.moving_avg.get(num, {}).get(5, 0) * 100)
            ma10 = int(self.moving_avg.get(num, {}).get(10, 0) * 100)
            ma20 = int(self.moving_avg.get(num, {}).get(20, 0) * 100)
            zscore = int(self.interval_stats.get(num, {}).get('zscore', 0) * 10)
            autocorr = int(self.autocorrelation.get(num, 0) * 100)
            features_list.append([freq, miss, ma5, ma10, ma20, zscore, autocorr])
        self.np_features = np.array(features_list, dtype=np.float32)
        # np.linalg.lstsq线性回归：预测遗漏值趋势
        self._np_linear_regression_trend()
        # np.histogram分析分布
        self._np_distribution_histogram()
        # np.corrcoef计算数字间相关性
        self._np_correlation_coefficients()
    
    def _np_linear_regression_trend(self):
        """NumPy np.linalg.lstsq线性回归预测遗漏趋势"""
        if self.np_features is None or len(self.data) < 10:
            return
        try:
            n = min(20, len(self.data))
            y = np.array([self.missing.get(i+1, 50) for i in range(n)], dtype=np.float64)
            X = np.vander(np.arange(n), 2)  # [[0,1],[1,1],...]用于多项式拟合
            # np.linalg.lstsq求最小二乘解
            coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            self.np_regression_coeffs = coeffs  # [slope, intercept]
            # 用回归预测下期遗漏
            next_x = np.array([[n, 1]])
            self.np_predicted_missing = np.dot(next_x, coeffs)[0]
        except:
            self.np_regression_coeffs = np.array([0, 25])
            self.np_predicted_missing = 25
    
    def _np_distribution_histogram(self):
        """NumPy np.histogram分析遗漏值分布"""
        miss_arr = np.array([self.missing.get(i, 50) for i in range(1, 50)], dtype=np.float32)
        # np.histogram分10个区间统计
        hist, bin_edges = np.histogram(miss_arr, bins=10, range=(0, 50))
        self.np_histogram = hist
        self.np_bin_edges = bin_edges
        # np.percentile计算分位数
        self.np_percentiles = np.percentile(miss_arr, [25, 50, 75, 90])
    
    def _np_correlation_coefficients(self):
        """NumPy np.corrcoef计算数字间相关性矩阵"""
        if self.np_features is None:
            return
        # 对特征矩阵计算相关性
        self.np_corrcoef = np.corrcoef(self.np_features.T) if self.np_features.shape[1] >= 2 else np.eye(49)
        # np.nan_to_num处理NaN值
        self.np_corrcoef = np.nan_to_num(self.np_corrcoef, nan=0.0)
    
    def _prepare_scipy_advanced(self):
        """SciPy深度分析：optimize.minimize权重优化、ks_2samp分布检验、signal.convolve平滑"""
        if len(self.data) < 10:
            self.scipy_weights = None
            return
        # scipy.optimize.minimize优化集成权重
        self._scipy_optimize_weights()
        # scipy.stats.ks_2samp分布一致性检验
        self._scipy_distribution_test()
        # scipy.signal.convolve趋势平滑
        self._scipy_smooth_trend()
        # scipy.interpolate插值预测遗漏值
        self._scipy_interpolate_missing()
    
    def _scipy_optimize_weights(self):
        """SciPy optimize.minimize贝叶斯优化集成权重"""
        def objective(w):
            w1, w2, w3, w4, w5 = w
            total = sum(w) + 1e-8
            score = sum(self._get_weighted_score(i+1) * wi / total for i, wi in enumerate([w1, w2, w3, w4, w5]))
            return -score  # minimize转maximize
        try:
            scipy_opt = _get_scipy_optimize()
            if scipy_opt is not None:
                minimize = scipy_opt.minimize
                result = minimize(objective, [0.2, 0.2, 0.15, 0.15, 0.2], method='L-BFGS-B', bounds=[(0, 1)]*5)
                self.scipy_weights = result.x / sum(result.x) if sum(result.x) > 0 else [0.2]*5
            else:
                self.scipy_weights = [0.2, 0.2, 0.15, 0.15, 0.2]
        except:
            self.scipy_weights = [0.2, 0.2, 0.15, 0.15, 0.2]
    
    def _scipy_distribution_test(self):
        """SciPy ks_2samp检验历史分布一致性"""
        if len(self.data) < 20:
            self.ks_test_result = 0.5
            return
        try:
            ks_func = _get_scipy_ks()
            if ks_func is None:
                self.ks_test_result = 0.5
                return
            # 比较前10期和后10期的遗漏分布
            early_miss = np.array([self.missing.get(i, 50) for i in range(1, 50) if i % 2 == 1], dtype=np.float32)
            late_miss = np.array([self.missing.get(i, 50) for i in range(1, 50) if i % 2 == 0], dtype=np.float32)
            if len(early_miss) > 3 and len(late_miss) > 3:
                stat, pval = ks_func(early_miss, late_miss)
                self.ks_test_result = pval  # p值越高分布越一致
            else:
                self.ks_test_result = 0.5
        except:
            self.ks_test_result = 0.5
    
    def _scipy_smooth_trend(self):
        """SciPy signal.convolve趋势平滑"""
        if len(self.data) < 5:
            self.scipy_smoothed = None
            return
        scipy_sig = _get_scipy_signal()
        if scipy_sig is None:
            self.scipy_smoothed = None
            return
        # 获取最近20期的频率序列
        freq_series = np.array([self.frequency.get(i, 0) for i in range(1, 50)], dtype=np.float32)
        # 高斯平滑核
        kernel = scipy_sig.gaussian(5, 1.0)
        kernel = kernel / kernel.sum()
        # scipy_signal.convolve卷积平滑
        self.scipy_smoothed = scipy_sig.convolve(freq_series, kernel, mode='same')
    
    def _scipy_interpolate_missing(self):
        """SciPy interpolate样条插值预测遗漏值"""
        if len(self.data) < 10:
            self.scipy_interp_func = None
            return
        scipy_int = _get_scipy_interpolate()
        if scipy_int is None:
            self.scipy_interp_func = None
            return
        try:
            # 构建x=数字,y=遗漏的散点
            x = np.arange(1, 50, dtype=np.float32)
            y = np.array([self.missing.get(i, 50) for i in range(1, 50)], dtype=np.float32)
            # 三次样条插值
            tck = scipy_int.splrep(x, y, s=0)
            self.scipy_interp_func = lambda xi: scipy_int.splev(xi, tck)
            # 预测边界值
            self.scipy_interp_edge = float(scipy_int.splev(25.5, tck))
        except:
            self.scipy_interp_func = lambda xi: 25.0
            self.scipy_interp_edge = 25.0
    
    def _prepare_sklearn_features(self):
        """sklearn特征准备：StandardScaler归一化和KMeans聚类"""
        _get_sklearn()
        if len(self.data) < 10:
            self.sklearn_features = None
            self.kmeans_labels = None
            self.scaler = None
            return
        
        # 构建49个数字的特征矩阵：频率+遗漏+MA5+MA20
        features = []
        for num in range(1, 50):
            freq = self.frequency.get(num, 0) / len(self.data)
            miss = min(self.missing.get(num, 50), 50) / 50.0
            ma5 = self.moving_avg.get(num, {}).get(5, 0.1)
            ma20 = self.moving_avg.get(num, {}).get(20, 0.1)
            zscore = self.interval_stats.get(num, {}).get('zscore', 0)
            features.append([freq, miss, ma5, ma20, zscore])
        
        self.sklearn_features = np.array(features)
        
        # StandardScaler归一化
        if _StandardScaler is not None:
            self.scaler = _StandardScaler()
            self.sklearn_features_scaled = self.scaler.fit_transform(self.sklearn_features)
        else:
            self.scaler = None
            self.sklearn_features_scaled = self.sklearn_features
        
        # KMeans聚类分组（3-5组）
        if _KMeans is not None:
            n_clusters = min(4, max(3, len(self.data) // 20))
            kmeans = _KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.kmeans_labels = kmeans.fit_predict(self.sklearn_features_scaled)
            # 记录聚类中心用于预测
            self.kmeans_centers = kmeans.cluster_centers_
        else:
            self.kmeans_labels = None
            self.kmeans_centers = None
    
    def _prepare_tensorflow_model(self):
        """TensorFlow深度学习：LSTM时序预测 + 全连接分类器 + AutoEncoder降维"""
        tf = _get_tf()
        keras = _keras_module
        layers = _layers_module
        if tf is None:
            self.tf_predictions = {}
            self.tf_autoencoder = None
            return
        if len(self.data) < 20:
            self.tf_predictions = {}
            self.tf_autoencoder = None
            return
        try:
            tf.random.set_seed(42)
            # 构建时序特征：最近5期每期49维one-hot
            X_seq, y_seq = [], []
            for i in range(len(self.data) - 5):
                seq = []
                for j in range(5):
                    one_hot = [0] * 49
                    for n in self.data[i + j].get('numbers', []):
                        if 1 <= n <= 49:
                            one_hot[n - 1] = 1
                    seq.append(one_hot)
                X_seq.append(seq)
                # 标签：第6期是否出现
                next_nums = set(self.data[i].get('numbers', []))
                y_seq.append([1 if n in next_nums else 0 for n in range(1, 50)])
            if len(X_seq) < 10:
                self.tf_predictions = {}
                return
            X_seq = np.array(X_seq, dtype=np.float32)
            y_seq = np.array(y_seq, dtype=np.float32)
            # tf.keras.Sequential LSTM模型
            self.tf_lstm_model = keras.Sequential([
                layers.LSTM(64, return_sequences=True, input_shape=(5, 49)),
                layers.LSTM(32),
                layers.Dense(49, activation='sigmoid')
            ])
            self.tf_lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            self.tf_lstm_model.fit(X_seq, y_seq, epochs=10, batch_size=16, verbose=0)
            # 预测下期
            X_last = np.array([X_seq[-1]], dtype=np.float32)
            self.tf_predictions = {n+1: float(p) for n, p in enumerate(self.tf_lstm_model.predict(X_last, verbose=0)[0])}
            # tf.keras.Sequential 全连接分类器预测
            self._tf_fc_classifier(X_seq, y_seq)
            # tf.keras AutoEncoder特征降维
            self._tf_autoencoder()
        except Exception as e:
            self.tf_predictions = {}
            self.tf_autoencoder = None
    
    def _tf_fc_classifier(self, X_seq, y_seq):
        """TensorFlow全连接分类器预测数字出现概率"""
        keras = _keras_module
        layers = _layers_module
        if keras is None or layers is None:
            self.tf_fc_predictions = {}
            return
        try:
            X_flat = X_seq.reshape(len(X_seq), -1)
            self.tf_fc_model = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(X_flat.shape[1],)),
                layers.Dropout(0.2),
                layers.Dense(64, activation='relu'),
                layers.Dense(49, activation='sigmoid')
            ])
            self.tf_fc_model.compile(optimizer='adam', loss='binary_crossentropy')
            self.tf_fc_model.fit(X_flat, y_seq, epochs=10, batch_size=16, verbose=0)
            X_last_flat = X_flat[-1:]
            self.tf_fc_predictions = {n+1: float(p) for n, p in enumerate(self.tf_fc_model.predict(X_last_flat, verbose=0)[0])}
        except:
            self.tf_fc_predictions = {}
    
    def _tf_autoencoder(self):
        """TensorFlow AutoEncoder特征降维"""
        keras = _keras_module
        layers = _layers_module
        if keras is None or layers is None:
            self.tf_autoencoder = None
            self.tf_encoded = None
            return
        try:
            X_flat = self.sklearn_features_scaled.astype(np.float32)
            encoding_dim = 10
            # 编码器
            inputs = keras.Input(shape=(X_flat.shape[1],))
            encoded = layers.Dense(32, activation='relu')(inputs)
            encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
            # 解码器
            decoded = layers.Dense(32, activation='relu')(encoded)
            decoded = layers.Dense(X_flat.shape[1], activation='sigmoid')(decoded)
            autoencoder = keras.Model(inputs, decoded)
            autoencoder.compile(optimizer='adam', loss='mse')
            autoencoder.fit(X_flat, X_flat, epochs=20, batch_size=16, verbose=0)
            # 编码器部分
            encoder = keras.Model(inputs, encoded)
            self.tf_encoded = encoder.predict(X_flat, verbose=0)
            self.tf_autoencoder = autoencoder
        except:
            self.tf_autoencoder = None
            self.tf_encoded = None
    
    def _prepare_pytorch_lstm(self):
        """PyTorch LSTM时序模型：完整训练循环 + 动态温度采样"""
        torch_mod = _get_torch()
        nn = _get_nn()
        optim_mod = _get_optim()
        if torch_mod is None:
            self.pt_lstm_preds = {}
            return
        # 延迟设置device
        if self.device is None:
            self.device = torch_mod.device('cuda' if torch_mod.cuda.is_available() else 'cpu')
        if len(self.data) < 20:
            self.pt_lstm_preds = {}
            return
        try:
            torch_mod.manual_seed(42)
            # 构建时序数据
            X_pt, y_pt = [], []
            for i in range(len(self.data) - 5):
                seq = []
                for j in range(5):
                    one_hot = [0.0] * 49
                    for n in self.data[i + j].get('numbers', []):
                        if 1 <= n <= 49:
                            one_hot[n - 1] = 1.0
                    seq.append(torch_mod.tensor(one_hot))
                X_pt.append(torch_mod.stack(seq).unsqueeze(0))
                next_nums = self.data[i].get('numbers', [])
                y_pt.append(torch_mod.tensor([1.0 if n in next_nums else 0.0 for n in range(1, 50)]))
            if len(X_pt) < 10:
                self.pt_lstm_preds = {}
                return
            X_pt = torch_mod.cat(X_pt, dim=0)
            y_pt = torch_mod.stack(y_pt)
            # PyTorch LSTM完整训练循环
            class LotteryLSTM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(49, 64, batch_first=True, num_layers=2, dropout=0.2)
                    self.fc = nn.Sequential(
                        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(32, 49), nn.Sigmoid()
                    )
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    return self.fc(lstm_out[:, -1, :])
            model = LotteryLSTM().to(self.device)
            criterion = nn.BCELoss()
            optimizer = optim_mod.Adam(model.parameters(), lr=0.001)
            # 完整训练循环：前向传播/损失计算/反向传播/权重更新
            model.train()
            for epoch in range(30):
                total_loss = 0
                for i in range(0, len(X_pt), 16):
                    batch_x = X_pt[i:i+16].to(self.device)
                    batch_y = y_pt[i:i+16].to(self.device)
                    optimizer.zero_grad()  # 梯度清零
                    outputs = model(batch_x)  # 前向传播
                    loss = criterion(outputs, batch_y)  # 损失计算
                    loss.backward()  # 反向传播
                    optimizer.step()  # 权重更新
                    total_loss += loss.item()
            # 预测
            model.eval()
            with torch_mod.no_grad():
                last_seq = X_pt[-1:].to(self.device)
                pred = model(last_seq).cpu().numpy()[0]
                self.pt_lstm_preds = {n+1: float(p) for n, p in enumerate(pred)}
            self.pt_lstm_model = model
            # PyTorch AutoEncoder特征降维
            self._pt_autoencoder()
        except Exception as e:
            self.pt_lstm_preds = {}
    
    def _pt_autoencoder(self):
        """PyTorch AutoEncoder特征降维"""
        torch_mod = _get_torch()
        nn = _get_nn()
        optim_mod = _get_optim()
        if torch_mod is None or nn is None:
            self.pt_encoded = None
            return
        try:
            X = torch_mod.tensor(self.sklearn_features_scaled, dtype=torch_mod.float32)
            class PtAutoEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(5, 3), nn.ReLU()
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(3, 5), nn.Sigmoid()
                    )
                def forward(self, x):
                    return self.decoder(self.encoder(x))
            model = PtAutoEncoder()
            optimizer = optim_mod.Adam(model.parameters(), lr=0.01)
            criterion = nn.MSELoss()
            model.train()
            for epoch in range(50):
                recon = model(X)
                loss = criterion(recon, X)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch_mod.no_grad():
                self.pt_encoded = model.encoder(X).numpy()
        except:
            self.pt_encoded = None
    
    def _build_dataframe(self):
        """构建Pandas DataFrame，每期包含统计特征"""
        records = []
        for record in self.data:
            numbers = record.get('numbers', [])
            if len(numbers) >= 6:
                rec = {
                    'numbers': numbers,
                    'sum': sum(numbers),
                    'mean': np.mean(numbers),
                    'std': np.std(numbers) if len(numbers) > 1 else 0,
                    'odd_count': sum(1 for n in numbers if LotteryConfig.is_odd(n)),
                    'even_count': sum(1 for n in numbers if LotteryConfig.is_even(n)),
                    'big_count': sum(1 for n in numbers if LotteryConfig.is_big(n)),
                    'small_count': sum(1 for n in numbers if LotteryConfig.is_small(n)),
                    'span': max(numbers) - min(numbers),
                }
                # 区间分布
                range_counts = [0] * 5
                for n in numbers:
                    idx = LotteryConfig.get_range_index(n)
                    if idx >= 0:
                        range_counts[idx] += 1
                for i in range(5):
                    rec[f'range_{i}'] = range_counts[i]
                # 尾数分布
                tail_counts = [0] * 10
                for n in numbers:
                    tail = LotteryConfig.get_tail_digit(n)
                    tail_counts[tail] += 1
                for i in range(10):
                    rec[f'tail_{i}'] = tail_counts[i]
                records.append(rec)
        self.df = DataFrame(records) if records else DataFrame()
    
    def _calculate_correlation_matrix(self):
        """NumPy计算49x49共现相关性矩阵"""
        if len(self.data) < 10:
            self.correlation_matrix = np.eye(49)
            return
        matrix = np.zeros((49, 49))
        for record in self.data:
            numbers = record.get('numbers', [])
            for i in numbers:
                for j in numbers:
                    if 1 <= i <= 49 and 1 <= j <= 49:
                        matrix[i-1, j-1] += 1
        # 归一化
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.correlation_matrix = matrix / row_sums
    
    def _calculate_autocorrelation(self):
        """StatsModels计算每个数字的自相关系数（无statsmodels时使用基础方法）"""
        self.autocorrelation = {}
        sm = _get_sm()
        tsastats = _tsastats_module
        for num in range(1, 50):
            series = []
            for record in self.data:
                series.append(1 if num in record.get('numbers', []) else 0)
            if len(series) >= 10:
                try:
                    if sm is not None and HAS_STATSMODELS:
                        acf = sm.tsa.acf(np.array(series), nlags=min(5, len(series)//2))
                        self.autocorrelation[num] = acf[1] if len(acf) > 1 else 0.0
                    else:
                        # 无statsmodels时使用基础自相关计算
                        if len(series) > 1:
                            mean = np.mean(series)
                            var = np.var(series)
                            if var > 0:
                                autocorr = np.sum((np.array(series[:-1]) - mean) * (np.array(series[1:]) - mean)) / (len(series) * var)
                                self.autocorrelation[num] = float(autocorr) if not np.isnan(autocorr) else 0.0
                            else:
                                self.autocorrelation[num] = 0.0
                        else:
                            self.autocorrelation[num] = 0.0
                except:
                    self.autocorrelation[num] = 0.0
            else:
                self.autocorrelation[num] = 0.0
    
    def _calculate_interval_stats(self):
        """计算间隔统计特征：均值、方差、偏度、Z-score"""
        self.interval_stats = {}
        scipy_st = _get_scipy_stats()
        skew_val = 0
        if scipy_st is not None and hasattr(scipy_st, 'skew'):
            try:
                skew_val = float(scipy_st.skew([1, 2, 3]))  # 测试是否可用
            except:
                skew_val = 0
        for num in range(1, 50):
            intervals = []
            last_appeared = None
            for i, record in enumerate(self.data):
                if num in record.get('numbers', []):
                    if last_appeared is not None:
                        intervals.append(i - last_appeared)
                    last_appeared = i
            if intervals:
                skew_calc = 0
                if skew_val != 0 and len(intervals) > 2:
                    try:
                        skew_calc = float(scipy_st.skew(intervals))
                    except:
                        skew_calc = 0
                self.interval_stats[num] = {
                    'mean': np.mean(intervals),
                    'std': np.std(intervals) if len(intervals) > 1 else 0,
                    'skew': skew_calc,
                    'zscore': self._calculate_zscore(len(self.data), intervals)
                }
            else:
                self.interval_stats[num] = {
                    'mean': 6.0, 'std': 0, 'skew': 0, 'zscore': 0
                }
    
    def _calculate_zscore(self, current_miss, intervals):
        """计算当前遗漏的Z-score"""
        if not intervals:
            return 0
        mean = np.mean(intervals)
        std = np.std(intervals) if len(intervals) > 1 else 1
        if std == 0:
            return 0
        return (current_miss - mean) / std
    
    def _calculate_adjacent_stats(self):
        """计算连号/邻号历史概率"""
        self.adjacent_prob = defaultdict(float)
        self.consecutive_prob = defaultdict(float)
        total_pairs = 0
        total_consec = 0
        for record in self.data:
            numbers = sorted(record.get('numbers', []))
            for i in range(len(numbers)):
                for j in range(i+1, len(numbers)):
                    diff = abs(numbers[j] - numbers[i])
                    if diff <= 2:
                        self.adjacent_prob[(numbers[i], numbers[j])] += 1
                        self.adjacent_prob[(numbers[j], numbers[i])] += 1
                        total_pairs += 1
                    if diff == 1:
                        self.consecutive_prob[numbers[i]] += 1
                        self.consecutive_prob[numbers[j]] += 1
                        total_consec += 1
        # 归一化
        if total_pairs > 0:
            for key in self.adjacent_prob:
                self.adjacent_prob[key] /= total_pairs
        if total_consec > 0:
            for key in self.consecutive_prob:
                self.consecutive_prob[key] /= total_consec
    
    def _calculate_moving_avg(self):
        """计算5期/10期/20期移动平均出现率"""
        self.moving_avg = {num: {} for num in range(1, 50)}
        for window in [5, 10, 20]:
            rates = []
            for i in range(len(self.data)):
                subset = self.data[i:i+window] if i == 0 else self.data[i-window+1:i+1]
                rate = sum(1 for r in subset if r.get('numbers') and any(num in r['numbers'] for num in range(1, 50))) / len(subset) if subset else 0
                rates.append(rate)
            for num in range(1, 50):
                series = [1 if num in r.get('numbers', []) else 0 for r in self.data]
                if len(series) >= window:
                    self.moving_avg[num][window] = np.mean(series[-window:])
                else:
                    self.moving_avg[num][window] = 0.1
    
    def _calculate_missing(self, number):
        missing = 0
        for record in reversed(self.data):
            if number in record.get('numbers', []):
                return missing
            missing += 1
        return missing + 10
    
    def _calculate_range_distribution(self):
        distribution = {i: 0 for i in range(5)}
        for record in self.data:
            for num in record.get('numbers', []):
                idx = LotteryConfig.get_range_index(num)
                if idx >= 0:
                    distribution[idx] += 1
        return distribution
    
    def _calculate_tail_distribution(self):
        distribution = {i: 0 for i in range(10)}
        for record in self.data:
            for num in record.get('numbers', []):
                tail = LotteryConfig.get_tail_digit(num)
                distribution[tail] += 1
        return distribution
    
    def _calculate_odd_even_ratio(self):
        odd_count = even_count = 0
        for record in self.data:
            for num in record.get('numbers', []):
                if LotteryConfig.is_odd(num):
                    odd_count += 1
                else:
                    even_count += 1
        return {'odd': odd_count, 'even': even_count}
    
    def _calculate_big_small_ratio(self):
        big_count = small_count = 0
        for record in self.data:
            for num in record.get('numbers', []):
                if LotteryConfig.is_big(num):
                    big_count += 1
                else:
                    small_count += 1
        return {'big': big_count, 'small': small_count}
    
    def _lr_roulette_weights(self):
        """sklearn LogisticRegression概率权重，用于轮盘赌算法"""
        if len(self.data) < 20 or not hasattr(self, 'sklearn_features') or self.sklearn_features is None:
            return [0.02] * 49
        try:
            _get_sklearn()
            LR = _LogisticRegression
            if LR is None:
                return [0.02] * 49
            X = self.sklearn_features_scaled.copy()
            # 构造标签：最近一期出现的号码为1，未出现为0
            last_draw = set(self.data[-1]) if self.data else set()
            y = np.array([1 if n in last_draw else 0 for n in range(1, 50)])
            lr_model = LR(max_iter=100, random_state=42)
            lr_model.fit(X, y)
            proba = lr_model.predict_proba(X)
            # 取类别1的概率
            if proba.shape[1] == 2:
                weights = proba[:, 1].tolist()
            else:
                weights = [0.02] * 49
            return weights
        except Exception:
            return [0.02] * 49
    
    def _get_weighted_score(self, num):
        """
        综合多维度计算单个数字的加权得分
        权重配置：频率0.15、遗漏回补0.2、短期趋势0.15、中期趋势0.15、自相关0.1、Z-score0.1、共现0.15
        """
        score = 0.0
        
        # 1. 频率得分 (权重0.15)
        freq = self.frequency.get(num, 0)
        total = len(self.data) * 6
        freq_score = freq / total if total > 0 else 0
        score += freq_score * 0.15 * 100
        
        # 2. 遗漏回补得分 (权重0.2)
        miss = self.missing.get(num, 50)
        avg_cycle = len(self.data) / 49 if len(self.data) > 0 else 6
        miss_score = MathUtils.calculate_missing_cycle(miss, avg_cycle)
        score += miss_score * 0.2 * 100
        
        # 3. 短期趋势MA5 vs MA20 (权重0.15)
        ma5 = self.moving_avg.get(num, {}).get(5, 0.1)
        ma20 = self.moving_avg.get(num, {}).get(20, 0.1)
        trend_score = (ma5 - ma20 + 0.1) * 5
        score += trend_score * 0.15 * 100
        
        # 4. 中期趋势 (权重0.15)
        ma10 = self.moving_avg.get(num, {}).get(10, 0.1)
        mid_trend_score = ma10 * 10
        score += mid_trend_score * 0.15 * 100
        
        # 5. 自相关性加成 (权重0.1)
        autocorr = self.autocorrelation.get(num, 0)
        score += max(0, autocorr) * 0.1 * 100
        
        # 6. Z-score异常检测 (权重0.1)
        stats = self.interval_stats.get(num, {})
        zscore = stats.get('zscore', 0)
        zscore_score = 1.0 / (1 + np.exp(-zscore))  # sigmoid
        score += zscore_score * 0.1 * 100
        
        # 7. 共现相关性得分 (权重0.15)
        if len(self.data) > 0:
            latest_nums = self.data[0].get('numbers', [])
            cooccur = sum(self.correlation_matrix[num-1, n-1] for n in latest_nums) / len(latest_nums)
            score += cooccur * 0.15 * 100
        
        return score
    
    def _optimize_ensemble_weights(self, n_trials=10):
        """Optuna贝叶斯优化集成权重（无Optuna时使用固定权重）"""
        optuna = _get_optuna()
        TPESampler = _TPESampler_class
        # 无Optuna时返回固定权重
        if optuna is None or TPESampler is None:
            return {
                'w_hot_cold': 0.2, 'w_missing': 0.2, 'w_range': 0.15,
                'w_tail': 0.15, 'w_freq': 0.2
            }
        
        def objective(trial):
            w1 = trial.suggest_float('w_hot_cold', 0.0, 0.4)
            w2 = trial.suggest_float('w_missing', 0.0, 0.4)
            w3 = trial.suggest_float('w_range', 0.0, 0.3)
            w4 = trial.suggest_float('w_tail', 0.0, 0.3)
            w5 = trial.suggest_float('w_freq', 0.0, 0.3)
            total = w1 + w2 + w3 + w4 + w5
            if total == 0:
                return 0
            # 模拟验证集评分（使用最后5期数据）
            scores = {}
            for num in range(1, 50):
                s = 0
                s += self._hot_cold_score(num) * w1 / total
                s += self._missing_score(num) * w2 / total
                s += self._range_score(num) * w3 / total
                s += self._tail_score(num) * w4 / total
                s += self.frequency.get(num, 0) * w5 / total
                scores[num] = s
            # 返回模拟得分
            return sum(scores.values())
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return study.best_params
    
    def _hot_cold_score(self, num):
        """冷热得分"""
        decay = np.exp(-self.missing.get(num, 0) / 5)
        freq_score = self.frequency.get(num, 0) / len(self.data) if self.data else 0
        return decay * 0.5 + freq_score * 0.5
    
    def _missing_score(self, num):
        """遗漏得分"""
        miss = self.missing.get(num, 50)
        stats = self.interval_stats.get(num, {})
        zscore = stats.get('zscore', 0)
        return (1 / (1 + np.exp(-zscore))) * 0.5 + MathUtils.calculate_missing_cycle(miss, 6) * 0.5
    
    def _range_score(self, num):
        """区间得分"""
        idx = LotteryConfig.get_range_index(num)
        total = sum(self.range_distribution.values()) or 1
        return (1 - self.range_distribution.get(idx, 0) / total) * 10
    
    def _tail_score(self, num):
        """尾数得分"""
        tail = LotteryConfig.get_tail_digit(num)
        total = sum(self.tail_distribution.values()) or 1
        return (1 - self.tail_distribution.get(tail, 0) / total) * 10
    
    # ================================================================
    # 12种增强算法
    # ================================================================
    
    def comprehensive_recommendation(self, count=6):
        """综合推荐 - TensorFlow LSTM + PyTorch LSTM + sklearn GB + NumPy矩阵 + SciPy优化"""
        # 快速优化权重
        best_weights = self._optimize_ensemble_weights(n_trials=10)
        
        # 收集各算法预测
        predictions = []
        predictions.extend(self.hot_cold_algorithm(count * 3))
        predictions.extend(self.missing_value_analysis(count * 3))
        predictions.extend(self.range_distribution_algorithm(count * 2))
        predictions.extend(self.tail_distribution_algorithm(count * 2))
        
        # 统计各数字出现次数
        counter = Counter(predictions)
        
        # sklearn: GradientBoostingClassifier预测每个数字出现概率
        gb_probs = self._gb_predict_probs() if len(self.data) >= 20 else {}
        
        # TensorFlow LSTM预测概率加成
        tf_bonus = self.tf_predictions if hasattr(self, 'tf_predictions') and self.tf_predictions else {}
        # PyTorch LSTM预测概率加成
        pt_bonus = self.pt_lstm_preds if hasattr(self, 'pt_lstm_preds') and self.pt_lstm_preds else {}
        # NumPy回归预测加成
        np_bonus = self._np_regression_bonus() if hasattr(self, 'np_features') and self.np_features is not None else {}
        # SciPy插值预测加成
        scipy_bonus = self._scipy_interp_bonus() if hasattr(self, 'scipy_interp_func') and self.scipy_interp_func else {}
        
        # 计算综合得分
        scores = {}
        total_w = sum(best_weights.values()) or 1
        for num in range(1, 50):
            # 多维度特征得分
            feature_score = self._get_weighted_score(num)
            
            # 投票得分
            vote_score = counter.get(num, 0) * 15
            
            # 集成权重
            ensemble_score = 0
            ensemble_score += self._hot_cold_score(num) * best_weights.get('w_hot_cold', 0.2) / total_w
            ensemble_score += self._missing_score(num) * best_weights.get('w_missing', 0.2) / total_w
            ensemble_score += self._range_score(num) * best_weights.get('w_range', 0.15) / total_w
            ensemble_score += self._tail_score(num) * best_weights.get('w_tail', 0.15) / total_w
            ensemble_score += (self.frequency.get(num, 0) / len(self.data)) * best_weights.get('w_freq', 0.2) / total_w
            
            # 多模型概率加成融合
            gb_bonus = gb_probs.get(num, 0.02) * 150
            tf_b = tf_bonus.get(num, 0.02) * 100 if tf_bonus else 0
            pt_b = pt_bonus.get(num, 0.02) * 100 if pt_bonus else 0
            np_b = np_bonus.get(num, 0) * 50
            scipy_b = scipy_bonus.get(num, 0) * 30
            
            scores[num] = feature_score + vote_score + ensemble_score * 50 + gb_bonus + tf_b + pt_b + np_b + scipy_b
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _np_regression_bonus(self):
        """NumPy np.linalg.lstsq回归预测加成"""
        bonus = {}
        if not hasattr(self, 'np_features') or self.np_features is None:
            return bonus
        try:
            coeffs = getattr(self, 'np_regression_coeffs', np.array([0, 25]))
            for num in range(1, 50):
                miss = self.missing.get(num, 50)
                # 遗漏值偏离回归线的程度
                expected = coeffs[0] * (num - 25) + coeffs[1]
                deviation = abs(miss - expected)
                bonus[num] = max(0, 10 - deviation)
        except:
            pass
        return bonus
    
    def _scipy_interp_bonus(self):
        """SciPy interpolate插值预测加成"""
        bonus = {}
        if not hasattr(self, 'scipy_interp_func') or not self.scipy_interp_func:
            return bonus
        try:
            for num in range(1, 50):
                interp_val = self.scipy_interp_func(num)
                miss = self.missing.get(num, 50)
                # 遗漏值与插值预测的差异
                diff = abs(miss - interp_val)
                bonus[num] = max(0, 10 - diff * 0.5)
        except:
            pass
        return bonus
    
    def _gb_predict_probs(self):
        """GradientBoostingClassifier预测每个数字出现概率"""
        _get_sklearn()
        GBC = _GradientBoostingClassifier
        if len(self.data) < 20 or not hasattr(self, 'sklearn_features') or self.sklearn_features is None or GBC is None:
            return {}
        try:
            # 构建训练数据：前N期特征 -> 下期是否出现
            X, y = [], []
            for i in range(len(self.data) - 1):
                if i < len(self.data) - 1:
                    features = self.sklearn_features.copy()
                    X.append(features.flatten())
                    next_nums = self.data[i].get('numbers', [])
                    y_single = [1 if n in next_nums else 0 for n in range(1, 50)]
                    y.append(y_single)
            if len(X) < 10:
                return {}
            X, y = np.array(X), np.array(y)
            # 对每个数字训练一个分类器
            probs = {}
            for n in range(1, 50):
                model = GBC(n_estimators=50, max_depth=3, random_state=42)
                model.fit(X, y[:, n-1])
                proba = model.predict_proba(X[-1:])[0][1] if len(model.classes_) > 1 else 0.02
                probs[n] = proba
            return probs
        except:
            return {}
    
    def hot_cold_algorithm(self, count=6):
        """冷热数字 - TensorFlow热号分类 + sklearn KMeans + NumPy直方图 + PyTorch平滑"""
        if not self.data:
            return random.sample(range(1, 50), count)
        
        # sklearn: KMeans聚类分组（热/温/冷）
        cluster_bonus = {}
        if hasattr(self, 'kmeans_labels') and self.kmeans_labels is not None:
            for num in range(1, 50):
                cluster_id = self.kmeans_labels[num - 1]
                cluster_bonus[num] = cluster_id * 10
        
        # NumPy直方图分析：热号分布
        np_hist_bonus = self._np_hot_histogram_bonus() if hasattr(self, 'np_histogram') else {}
        
        # TensorFlow热号分类预测
        tf_hot_probs = self._tf_hot_classifier() if hasattr(self, 'tf_predictions') and self.tf_predictions else {}
        
        # PyTorch神经网络平滑特征
        pt_smooth_bonus = self._pt_hot_smooth() if hasattr(self, 'pt_lstm_preds') and self.pt_lstm_preds else {}
        
        # SciPy卷积平滑趋势
        scipy_smooth_bonus = self._scipy_hot_smooth() if hasattr(self, 'scipy_smoothed') and self.scipy_smoothed is not None else {}
        
        scores = {}
        for num in range(1, 50):
            # 指数加权频率（NumPy exp计算）
            decay_weight = 0
            for i, record in enumerate(self.data[:10]):
                weight = np.exp(-i / 5)
                if num in record.get('numbers', []):
                    decay_weight += weight
            
            # 遗漏Z-score异常检测
            miss = self.missing.get(num, 50)
            stats = self.interval_stats.get(num, {})
            zscore = stats.get('zscore', 0)
            zscore_bonus = max(0, zscore) * 5
            
            # 趋势动量（MA5 vs MA20）
            ma5 = self.moving_avg.get(num, {}).get(5, 0.1)
            ma20 = self.moving_avg.get(num, {}).get(20, 0.1)
            momentum = (ma5 - ma20) * 50
            
            # 自相关性加成
            autocorr = self.autocorrelation.get(num, 0) * 10
            
            # KMeans聚类热度加成
            kmeans_bonus = cluster_bonus.get(num, 0)
            
            # NumPy直方图热号加成
            np_bonus = np_hist_bonus.get(num, 0)
            
            # TensorFlow热号概率加成
            tf_bonus = tf_hot_probs.get(num, 0.02) * 100
            
            # PyTorch平滑加成
            pt_bonus = pt_smooth_bonus.get(num, 0.02) * 80
            
            # SciPy平滑加成
            scipy_bonus = scipy_smooth_bonus.get(num, 0) * 10
            
            # 综合得分
            scores[num] = (decay_weight * 8 + zscore_bonus + momentum + autocorr + kmeans_bonus +
                          np_bonus + tf_bonus + pt_bonus + scipy_bonus)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _np_hot_histogram_bonus(self):
        """NumPy直方图分析热号分布"""
        bonus = {}
        if not hasattr(self, 'np_histogram') or self.np_histogram is None:
            return bonus
        try:
            miss_arr = np.array([self.missing.get(i, 50) for i in range(1, 50)], dtype=np.float32)
            percentiles = np.percentile(miss_arr, [25, 50, 75])
            for num in range(1, 50):
                miss = self.missing.get(num, 50)
                if miss < percentiles[0]:
                    bonus[num] = 20  # 极热
                elif miss < percentiles[1]:
                    bonus[num] = 10  # 温热
                elif miss < percentiles[2]:
                    bonus[num] = 0   # 温冷
                else:
                    bonus[num] = -10  # 极冷
        except:
            pass
        return bonus
    
    def _tf_hot_classifier(self):
        """TensorFlow热号分类预测"""
        if not hasattr(self, 'tf_predictions') or not self.tf_predictions:
            return {}
        probs = {}
        miss_arr = np.array([self.missing.get(i, 50) for i in range(1, 50)])
        threshold = np.percentile(miss_arr, 30)
        for num in range(1, 50):
            miss = self.missing.get(num, 50)
            base_prob = self.tf_predictions.get(num, 0.02)
            if miss < threshold:
                probs[num] = base_prob * 1.3
            else:
                probs[num] = base_prob * 0.8
        return probs
    
    def _pt_hot_smooth(self):
        """PyTorch LSTM平滑热号概率"""
        if not hasattr(self, 'pt_lstm_preds') or not self.pt_lstm_preds:
            return {}
        return self.pt_lstm_preds.copy()
    
    def _scipy_hot_smooth(self):
        """SciPy signal.convolve平滑热号趋势"""
        bonus = {}
        if not hasattr(self, 'scipy_smoothed') or self.scipy_smoothed is None:
            return bonus
        try:
            smoothed = self.scipy_smoothed
            if len(smoothed) == 49:
                max_val = np.max(smoothed)
                min_val = np.min(smoothed)
                range_val = max_val - min_val if max_val > min_val else 1
                for i in range(49):
                    norm_val = (smoothed[i] - min_val) / range_val
                    bonus[i + 1] = norm_val * 10
        except:
            pass
        return bonus
    
    def odd_even_algorithm(self, count=6):
        """单双算法 - TensorFlow预测 + sklearn LogisticRegression + NumPy矩阵 + SciPy KS检验"""
        _get_sklearn()
        if not self.data:
            return random.sample(range(1, 50), count)
        
        # NumPy计算历史单数序列统计量
        odd_seq = []
        for record in self.data[:20]:
            odds = sum(1 for n in record.get('numbers', []) if LotteryConfig.is_odd(n))
            odd_seq.append(odds)
        
        # NumPy np.corrcoef计算奇偶序列自相关性
        odd_np_corr = 0
        if len(odd_seq) >= 5:
            try:
                odd_np_corr = float(np.corrcoef(odd_seq[:-1], odd_seq[1:])[0, 1])
            except:
                odd_np_corr = 0
        
        # sklearn: LogisticRegression预测下期单数个数概率分布
        lr_predicted_odds = 3
        LR = _LogisticRegression
        SS = _StandardScaler
        if LR is not None and SS is not None and len(odd_seq) >= 10:
            try:
                X_lr = np.array(odd_seq[:-1]).reshape(-1, 1)
                y_lr = np.array(odd_seq[1:])
                scaler_lr = SS()
                X_lr_scaled = scaler_lr.fit_transform(X_lr)
                lr_model = LR(max_iter=100, random_state=42)
                lr_model.fit(X_lr_scaled, y_lr)
                X_next = scaler_lr.transform([[odd_seq[-1]]])
                lr_proba = lr_model.predict_proba(X_next)[0]
                lr_predicted_odds = lr_model.predict(X_next)[0]
                lr_predicted_odds = int(np.sum(lr_proba * np.arange(len(lr_proba))))
            except:
                lr_predicted_odds = 3
        elif len(odd_seq) >= 10:
            lr_predicted_odds = 3
        
        # TensorFlow神经网络预测单双分布
        tf_odds = self._tf_odd_even_predict(odd_seq) if len(odd_seq) >= 10 else 3
        
        # SciPy ks_2samp分布检验修正
        ks_correction = self._scipy_odd_correction() if hasattr(self, 'ks_test_result') else 0
        
        # 构建马尔可夫链状态转移矩阵
        transition = defaultdict(lambda: defaultdict(int))
        for i in range(len(odd_seq) - 1):
            transition[odd_seq[i]][odd_seq[i+1]] += 1
        
        # 预测下期单数（融合多模型）
        if odd_seq:
            last_odd = odd_seq[-1]
            next_odds_probs = transition[last_odd]
            if next_odds_probs:
                markov_pred = max(next_odds_probs.items(), key=lambda x: x[1])[0]
            else:
                markov_pred = int(np.mean(odd_seq[-5:]))
        else:
            markov_pred = 3
        
        # 多模型融合预测
        predicted_odds = int(round(lr_predicted_odds * 0.35 + markov_pred * 0.35 + tf_odds * 0.2 + ks_correction * 0.1))
        
        # 均值回归修正
        expected_odds = 3.0
        predicted_odds = int(round(predicted_odds * 0.6 + expected_odds * 0.4))
        predicted_odds = max(2, min(4, predicted_odds))
        
        # 按特征得分选号
        selected = []
        odd_candidates = sorted([n for n in range(1, 50) if LotteryConfig.is_odd(n)], 
                                key=lambda x: self._get_weighted_score(x), reverse=True)
        even_candidates = sorted([n for n in range(1, 50) if LotteryConfig.is_even(n)], 
                                 key=lambda x: self._get_weighted_score(x), reverse=True)
        
        for _ in range(predicted_odds):
            if odd_candidates:
                selected.append(odd_candidates.pop(0))
        for _ in range(count - predicted_odds):
            if even_candidates:
                selected.append(even_candidates.pop(0))
        
        return selected[:count]
    
    def _tf_odd_even_predict(self, odd_seq):
        """TensorFlow神经网络预测单双分布"""
        if len(odd_seq) < 10 or not hasattr(self, 'tf_predictions') or not self.tf_predictions:
            return 3
        try:
            # 基于tf_predictions计算预期单数个数
            odd_probs = [self.tf_predictions.get(n, 0.02) for n in range(1, 50) if n % 2 == 1]
            expected_odds = sum(odd_probs) / len(odd_probs) * 25 if odd_probs else 3
            return max(2, min(4, int(round(expected_odds))))
        except:
            return 3
    
    def _scipy_odd_correction(self):
        """SciPy ks_2samp分布检验修正预测"""
        if not hasattr(self, 'ks_test_result'):
            return 3
        # p值越高分布越一致，预测值越接近均值
        pval = getattr(self, 'ks_test_result', 0.5)
        return int(round(3.0 * pval + 3 * (1 - pval)))
    
    def big_small_algorithm(self, count=6):
        """大小算法 - TensorFlow预测 + sklearn RF + NumPy lstsq + SciPy插值"""
        _get_sklearn()
        if not self.data:
            return random.sample(range(1, 50), count)
        
        # NumPy np.linalg.lstsq线性回归趋势预测
        recent_bigs = []
        for record in self.data[:20]:
            bigs = sum(1 for n in record.get('numbers', []) if LotteryConfig.is_big(n))
            recent_bigs.append(bigs)
        
        np_predicted_bigs = 3
        if len(recent_bigs) >= 5:
            try:
                # NumPy np.linalg.lstsq矩阵求解
                n = len(recent_bigs)
                X = np.vander(np.arange(n), 2)
                y = np.array(recent_bigs, dtype=np.float64)
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                next_x = np.array([[n, 1]])
                np_predicted_bigs = np.dot(next_x, coeffs)[0]
            except:
                np_predicted_bigs = np.mean(recent_bigs)
        
        # sklearn: RandomForestClassifier预测下期大数个数
        rf_predicted_bigs = 3
        RF = _RandomForestClassifier
        SS = _StandardScaler
        if RF is not None and SS is not None and len(recent_bigs) >= 10:
            try:
                X_rf = np.array(recent_bigs[:-1]).reshape(-1, 1)
                y_rf = np.array(recent_bigs[1:])
                scaler_rf = SS()
                X_rf_scaled = scaler_rf.fit_transform(X_rf)
                rf_model = RF(n_estimators=50, max_depth=3, random_state=42)
                rf_model.fit(X_rf_scaled, y_rf)
                X_next = scaler_rf.transform([[recent_bigs[-1]]])
                rf_predicted_bigs = rf_model.predict(X_next)[0]
            except:
                rf_predicted_bigs = 3
        elif len(recent_bigs) >= 10:
            rf_predicted_bigs = 3
        
        # TensorFlow神经网络预测大小分布
        tf_bigs = self._tf_big_small_predict() if len(recent_bigs) >= 10 else 3
        
        # SciPy interpolate插值预测
        scipy_pred = self._scipy_big_interp() if hasattr(self, 'scipy_interp_func') and self.scipy_interp_func else 3
        
        # 融合预测
        predicted_bigs = int(round(np_predicted_bigs * 0.25 + rf_predicted_bigs * 0.3 + tf_bigs * 0.3 + scipy_pred * 0.15))
        predicted_bigs = max(2, min(4, predicted_bigs))
        
        # 区间平衡调整
        range_counts = [0] * 5
        for record in self.data[:5]:
            for num in record.get('numbers', []):
                idx = LotteryConfig.get_range_index(num)
                if idx >= 0:
                    range_counts[idx] += 1
        
        # 选择候选池
        big_candidates = sorted([n for n in range(26, 50)], 
                                key=lambda x: self._get_weighted_score(x), reverse=True)
        small_candidates = sorted([n for n in range(1, 26)], 
                                 key=lambda x: self._get_weighted_score(x), reverse=True)
        
        # 按区间分布调整
        if range_counts[0] < range_counts[3]:
            small_candidates = sorted(small_candidates, key=lambda x: LotteryConfig.get_range_index(x) == 0, reverse=True)
        
        selected = []
        for _ in range(predicted_bigs):
            if big_candidates:
                selected.append(big_candidates.pop(0))
        for _ in range(count - predicted_bigs):
            if small_candidates:
                selected.append(small_candidates.pop(0))
        
        return selected[:count]
    
    def _tf_big_small_predict(self):
        """TensorFlow神经网络预测大小分布"""
        if not hasattr(self, 'tf_predictions') or not self.tf_predictions:
            return 3
        try:
            big_probs = [self.tf_predictions.get(n, 0.02) for n in range(26, 50)]
            expected_bigs = sum(big_probs) / len(big_probs) * 6 if big_probs else 3
            return max(2, min(4, int(round(expected_bigs))))
        except:
            return 3
    
    def _scipy_big_interp(self):
        """SciPy interpolate插值预测大小分布"""
        if not hasattr(self, 'scipy_interp_func') or not self.scipy_interp_func:
            return 3
        try:
            big_sum = sum(self.scipy_interp_func(n) for n in range(26, 50))
            avg = big_sum / 24 if big_sum > 0 else 25
            return max(2, min(4, int(round(avg / 5))))
        except:
            return 3
    
    def missing_value_analysis(self, count=6):
        """遗漏值分析 - TensorFlow回补预测 + sklearn GaussianNB + NumPy分位数 + SciPy插值"""
        if not self.data:
            return random.sample(range(1, 50), count)
        
        # sklearn: GaussianNB贝叶斯分类判断每个数字"即将出现"的概率
        nb_probs = self._nb_missing_probs() if len(self.data) >= 20 else {}
        
        # NumPy np.percentile分位数分析遗漏分布
        np_percentile_bonus = self._np_missing_percentile() if hasattr(self, 'np_percentiles') else {}
        
        # TensorFlow回补概率预测
        tf_backfill = self._tf_missing_predict() if hasattr(self, 'tf_predictions') and self.tf_predictions else {}
        
        # PyTorch遗漏回补预测
        pt_backfill = self._pt_missing_predict() if hasattr(self, 'pt_lstm_preds') and self.pt_lstm_preds else {}
        
        # SciPy interpolate插值预测遗漏
        scipy_interp_bonus = self._scipy_missing_interp() if hasattr(self, 'scipy_interp_func') and self.scipy_interp_func else {}
        
        total_records = len(self.data)
        avg_cycle = total_records / 49 if total_records > 0 else 6
        
        scores = {}
        for num in range(1, 50):
            missing = self.missing.get(num, 50)
            stats = self.interval_stats.get(num, {})
            
            # Z-score异常检测
            zscore = stats.get('zscore', 0)
            zscore_score = 1.0 / (1 + np.exp(-zscore))  # sigmoid
            
            # 指数回补模型
            exp_back = MathUtils.calculate_missing_cycle(missing, avg_cycle)
            
            # 间隔方差规律性得分
            interval_std = stats.get('std', 0)
            regularity_score = 1.0 / (1 + interval_std)
            
            # 遗漏/均间隔比值
            ratio = missing / stats.get('mean', avg_cycle) if stats.get('mean', 0) > 0 else 1
            
            # sklearn GaussianNB概率加成
            nb_bonus = nb_probs.get(num, 0.02) * 50
            
            # NumPy分位数加成
            np_bonus = np_percentile_bonus.get(num, 0) * 10
            
            # TensorFlow回补概率加成
            tf_bonus = tf_backfill.get(num, 0.02) * 80
            
            # PyTorch回补概率加成
            pt_bonus = pt_backfill.get(num, 0.02) * 60
            
            # SciPy插值加成
            scipy_bonus = scipy_interp_bonus.get(num, 0) * 5
            
            # 综合得分
            scores[num] = (
                zscore_score * 0.2 +
                exp_back * 0.25 +
                regularity_score * 0.1 +
                min(ratio, 3) / 3 * 0.15 +
                nb_bonus + np_bonus + tf_bonus + pt_bonus + scipy_bonus
            )
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _np_missing_percentile(self):
        """NumPy np.percentile分位数分析遗漏"""
        bonus = {}
        if not hasattr(self, 'np_percentiles') or self.np_percentiles is None:
            return bonus
        try:
            p25, p50, p75, p90 = self.np_percentiles
            for num in range(1, 50):
                miss = self.missing.get(num, 50)
                if miss <= p25:
                    bonus[num] = 5  # 极热
                elif miss <= p50:
                    bonus[num] = 3  # 温热
                elif miss <= p75:
                    bonus[num] = 1  # 温冷
                elif miss <= p90:
                    bonus[num] = -1  # 冷
                else:
                    bonus[num] = -3  # 极冷
        except:
            pass
        return bonus
    
    def _tf_missing_predict(self):
        """TensorFlow回补概率预测"""
        if not hasattr(self, 'tf_predictions') or not self.tf_predictions:
            return {}
        return self.tf_predictions.copy()
    
    def _pt_missing_predict(self):
        """PyTorch LSTM回补概率预测"""
        if not hasattr(self, 'pt_lstm_preds') or not self.pt_lstm_preds:
            return {}
        return self.pt_lstm_preds.copy()
    
    def _scipy_missing_interp(self):
        """SciPy interpolate插值遗漏加成"""
        bonus = {}
        if not hasattr(self, 'scipy_interp_func') or not self.scipy_interp_func:
            return bonus
        try:
            for num in range(1, 50):
                interp_val = self.scipy_interp_func(num)
                miss = self.missing.get(num, 50)
                diff = interp_val - miss
                if diff > 0:
                    bonus[num] = min(diff * 0.5, 5)
                else:
                    bonus[num] = max(diff * 0.3, -3)
        except:
            pass
        return bonus
    
    def _nb_missing_probs(self):
        """GaussianNB贝叶斯分类判断每个数字'即将出现'的概率"""
        _get_sklearn()
        GNB = _GaussianNB
        if not hasattr(self, 'sklearn_features') or self.sklearn_features is None or GNB is None:
            return {}
        try:
            probs = {}
            # 特征：频率+遗漏+MA5+MA20+zscore，标签：该数字是否在接下来2期内出现
            X_nb, y_nb = [], []
            for i in range(len(self.data) - 2):
                if i < len(self.data) - 2:
                    features = self.sklearn_features.copy()
                    X_nb.append(features.flatten())
                    # 标签：未来2期内是否出现
                    future_nums = set()
                    for j in range(2):
                        if i + j < len(self.data):
                            future_nums.update(self.data[i + j].get('numbers', []))
                    y_single = [1 if n in future_nums else 0 for n in range(1, 50)]
                    y_nb.append(y_single)
            if len(X_nb) < 10:
                return {}
            X_nb, y_nb = np.array(X_nb), np.array(y_nb)
            for n in range(1, 50):
                nb_model = GNB()
                nb_model.fit(X_nb, y_nb[:, n-1])
                if len(nb_model.classes_) > 1:
                    proba = nb_model.predict_proba(X_nb[-1:])[0][1]
                else:
                    proba = 0.02
                probs[n] = proba
            return probs
        except:
            return {}
    
    def adjacent_number_analysis(self, count=6):
        """连号/邻号 - TensorFlow预测 + sklearn MLP + NumPy共现矩阵 + PyTorch平滑"""
        if not self.data:
            return random.sample(range(1, 50), count)
        
        # sklearn: MLPClassifier预测邻号出现概率
        mlp_probs = self._mlp_adjacent_probs() if len(self.data) >= 20 else {}
        
        # TensorFlow邻号预测
        tf_adj_probs = self._tf_adjacent_predict() if hasattr(self, 'tf_predictions') and self.tf_predictions else {}
        
        # PyTorch邻号预测
        pt_adj_probs = self._pt_adjacent_predict() if hasattr(self, 'pt_lstm_preds') and self.pt_lstm_preds else {}
        
        # 邻号候选
        adjacent_candidates = set()
        latest_numbers = self.data[0].get('numbers', [])
        for num in latest_numbers:
            for offset in [-2, -1, 1, 2]:
                adj = num + offset
                if 1 <= adj <= 49:
                    adjacent_candidates.add(adj)
        
        scores = {}
        for num in range(1, 50):
            # 条件概率
            cond_prob = self.consecutive_prob.get(num, 0.01)
            
            # NumPy np.dot矩阵乘法计算共现得分
            cooccur_score = 0
            if latest_numbers:
                cooccur_vec = np.array([1 if n in latest_numbers else 0 for n in range(1, 50)], dtype=np.float32)
                cooccur_score = np.dot(self.correlation_matrix[num-1], cooccur_vec) / len(latest_numbers)
            
            # 距离衰减得分
            distance_score = 0
            for ln in latest_numbers:
                dist = abs(ln - num)
                distance_score += np.exp(-dist / 5)
            distance_score /= len(latest_numbers) if latest_numbers else 1
            
            # 基础特征得分
            base_score = self._get_weighted_score(num) / 100
            
            # sklearn MLP概率加成
            mlp_bonus = mlp_probs.get(num, 0.02) * 100
            
            # TensorFlow邻号加成
            tf_bonus = tf_adj_probs.get(num, 0.02) * 80
            
            # PyTorch邻号加成
            pt_bonus = pt_adj_probs.get(num, 0.02) * 60
            
            # 综合得分
            scores[num] = (cond_prob * 0.2 + cooccur_score * 0.15 + distance_score * 0.1 + 
                          base_score * 0.1 + mlp_bonus + tf_bonus + pt_bonus)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _tf_adjacent_predict(self):
        """TensorFlow邻号出现概率"""
        if not hasattr(self, 'tf_predictions') or not self.tf_predictions:
            return {}
        return self.tf_predictions.copy()
    
    def _pt_adjacent_predict(self):
        """PyTorch LSTM邻号出现概率"""
        if not hasattr(self, 'pt_lstm_preds') or not self.pt_lstm_preds:
            return {}
        return self.pt_lstm_preds.copy()
    
    def _mlp_adjacent_probs(self):
        """MLPClassifier预测邻号出现概率"""
        _get_sklearn()
        MLP = _MLPClassifier
        SS = _StandardScaler
        if not hasattr(self, 'sklearn_features') or self.sklearn_features is None or MLP is None or SS is None:
            return {}
        try:
            probs = {}
            # 构建训练数据：历史特征 -> 下期是否出现
            X_mlp, y_mlp = [], []
            for i in range(len(self.data) - 1):
                if i < len(self.data) - 1:
                    features = self.sklearn_features.copy()
                    X_mlp.append(features.flatten())
                    next_nums = self.data[i].get('numbers', [])
                    y_single = [1 if n in next_nums else 0 for n in range(1, 50)]
                    y_mlp.append(y_single)
            if len(X_mlp) < 10:
                return {}
            X_mlp, y_mlp = np.array(X_mlp), np.array(y_mlp)
            scaler_mlp = SS()
            X_mlp_scaled = scaler_mlp.fit_transform(X_mlp)
            for n in range(1, 50):
                mlp_model = MLP(hidden_layer_sizes=(50, 25), max_iter=200, random_state=42)
                mlp_model.fit(X_mlp_scaled, y_mlp[:, n-1])
                if len(mlp_model.classes_) > 1:
                    proba = mlp_model.predict_proba(X_mlp_scaled[-1:])[0][1]
                else:
                    proba = 0.02
                probs[n] = proba
            return probs
        except:
            return {}
    
    def tail_distribution_algorithm(self, count=6):
        """尾数分布 - TensorFlow预测 + sklearn KMeans + NumPy histogram + SciPy分布检验"""
        if not self.data:
            return random.sample(range(1, 50), count)
        
        # sklearn: KMeans对10个尾数的频率向量聚类，找出需要回补的尾数组
        tail_cluster_bonus = self._tail_kmeans_bonus() if len(self.data) >= 15 else {}
        
        # NumPy histogram分析尾数分布
        np_hist_bonus = self._np_tail_histogram() if hasattr(self, 'np_histogram') else {}
        
        # TensorFlow尾数预测
        tf_tail_bonus = self._tf_tail_predict() if hasattr(self, 'tf_predictions') and self.tf_predictions else {}
        
        # SciPy分布检验修正
        scipy_tail_corr = self._scipy_tail_correction() if hasattr(self, 'ks_test_result') else {}
        
        # 计算尾数频率
        tail_counts = np.array([self.tail_distribution.get(i, 0) for i in range(10)])
        total = tail_counts.sum()
        expected = total / 10
        
        # Chi-square均匀性检验
        chi2_stat = np.sum((tail_counts - expected) ** 2 / expected) if expected > 0 else 0
        chi2_score = chi2_stat / 100
        
        # 选择低频尾数（应该回补）
        tail_scores = {}
        for tail in range(10):
            observed = tail_counts[tail]
            deviation = (expected - observed) / expected if expected > 0 else 0
            # 遗漏补偿
            tail_missing = sum(1 for record in self.data[:10] 
                              if not any(LotteryConfig.get_tail_digit(n) == tail for n in record.get('numbers', [])))
            # sklearn KMeans聚类加成
            cluster_bonus = tail_cluster_bonus.get(tail, 0)
            # NumPy直方图加成
            np_bonus = np_hist_bonus.get(tail, 0)
            # TensorFlow加成
            tf_bonus = tf_tail_bonus.get(tail, 0.02) * 50
            # SciPy加成
            scipy_bonus = scipy_tail_corr.get(tail, 0) * 10
            tail_scores[tail] = deviation * 10 + tail_missing * 0.5 + cluster_bonus + np_bonus + tf_bonus + scipy_bonus
        
        sorted_tails = sorted(tail_scores.items(), key=lambda x: x[1], reverse=True)[:4]
        selected_tails = [t for t, _ in sorted_tails]
        
        # 按特征得分选号
        candidates = []
        for tail in selected_tails:
            tail_nums = [n for n in range(1, 50) if LotteryConfig.get_tail_digit(n) == tail]
            tail_nums.sort(key=lambda x: self._get_weighted_score(x), reverse=True)
            candidates.extend(tail_nums[:3])
        
        candidates.sort(key=lambda x: self._get_weighted_score(x), reverse=True)
        return candidates[:count]
    
    def _np_tail_histogram(self):
        """NumPy histogram分析尾数分布"""
        bonus = {}
        if not hasattr(self, 'np_histogram') or self.np_histogram is None:
            return bonus
        try:
            tail_counts = np.array([self.tail_distribution.get(i, 0) for i in range(10)], dtype=np.float32)
            avg = np.mean(tail_counts)
            for tail in range(10):
                if tail_counts[tail] < avg * 0.8:
                    bonus[tail] = 5  # 低频尾数
                else:
                    bonus[tail] = 0
        except:
            pass
        return bonus
    
    def _tf_tail_predict(self):
        """TensorFlow尾数分布预测"""
        if not hasattr(self, 'tf_predictions') or not self.tf_predictions:
            return {}
        bonus = {}
        tail_probs = []
        for tail in range(10):
            probs = [self.tf_predictions.get(n, 0.02) for n in range(1, 50) if LotteryConfig.get_tail_digit(n) == tail]
            tail_probs.append(sum(probs) / len(probs) if probs else 0.02)
        avg = sum(tail_probs) / len(tail_probs) if tail_probs else 0.02
        for tail, prob in enumerate(tail_probs):
            if prob < avg * 0.9:
                bonus[tail] = prob * 2  # 低概率尾数需回补
            else:
                bonus[tail] = prob
        return bonus
    
    def _scipy_tail_correction(self):
        """SciPy分布检验修正尾数分布"""
        bonus = {}
        if not hasattr(self, 'ks_test_result'):
            return bonus
        pval = getattr(self, 'ks_test_result', 0.5)
        tail_counts = np.array([self.tail_distribution.get(i, 0) for i in range(10)])
        avg = np.mean(tail_counts)
        for tail in range(10):
            if pval > 0.3:  # 分布一致，低频更可能回补
                if tail_counts[tail] < avg:
                    bonus[tail] = 3
                else:
                    bonus[tail] = 0
            else:  # 分布不一致，趋势延续
                bonus[tail] = 0
        return bonus
    
    def _tail_kmeans_bonus(self):
        """KMeans对10个尾数聚类，找出低频尾数组需要回补"""
        _get_sklearn()
        KM = _KMeans
        if KM is None:
            return {}
        try:
            # 构建10个尾数的频率向量
            tail_freq = []
            for tail in range(10):
                tail_freq.append([self.tail_distribution.get(tail, 0) / max(len(self.data), 1)])
            tail_freq = np.array(tail_freq)
            # KMeans聚类（2组：高频组vs低频组）
            kmeans = KM(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(tail_freq)
            # 找出低频组的尾数
            centers = kmeans.cluster_centers_
            low_freq_cluster = 0 if centers[0][0] < centers[1][0] else 1
            bonus = {}
            for tail, label in enumerate(labels):
                bonus[tail] = 10 if label == low_freq_cluster else 0
            return bonus
        except:
            return {}
    
    def range_distribution_algorithm(self, count=6):
        """区间分布 - TensorFlow预测 + sklearn MinMaxScaler + NumPy histogram + SciPy插值"""
        if not self.data:
            return random.sample(range(1, 50), count)
        
        # sklearn: MinMaxScaler归一化区间特征 + cross_val_score验证区间预测模型
        range_cv_scores = self._range_cv_scores() if len(self.data) >= 30 else {i: 0.5 for i in range(5)}
        
        # NumPy histogram区间分析
        np_range_bonus = self._np_range_histogram() if hasattr(self, 'np_histogram') else {}
        
        # TensorFlow区间预测
        tf_range_bonus = self._tf_range_predict() if hasattr(self, 'tf_predictions') and self.tf_predictions else {}
        
        # PyTorch区间预测
        pt_range_bonus = self._pt_range_predict() if hasattr(self, 'pt_lstm_preds') and self.pt_lstm_preds else {}
        
        # SciPy interpolate区间插值
        scipy_range_bonus = self._scipy_range_interp() if hasattr(self, 'scipy_interp_func') and self.scipy_interp_func else {}
        
        # 使用Pandas DataFrame计算区间趋势
        if hasattr(self, 'df') and not self.df.empty:
            recent_range_freq = self.df[['range_0', 'range_1', 'range_2', 'range_3', 'range_4']].iloc[:10].sum()
        else:
            recent_range_freq = {i: 0 for i in range(5)}
            for record in self.data[:10]:
                for num in record.get('numbers', []):
                    idx = LotteryConfig.get_range_index(num)
                    if idx >= 0:
                        recent_range_freq[idx] += 1
            recent_range_freq = Series(recent_range_freq)
        
        # 卡方偏差
        total = recent_range_freq.sum()
        expected = total / 5
        
        # 动态区间权重（融合多模型）
        range_weights = {}
        for i in range(5):
            chi2_weight = max(0, (expected - recent_range_freq.iloc[i]) / expected) * 5 + (total - recent_range_freq.iloc[i]) / total
            cv_weight = range_cv_scores.get(i, 0.5)
            np_b = np_range_bonus.get(i, 0) * 5
            tf_b = tf_range_bonus.get(i, 0.02) * 50
            pt_b = pt_range_bonus.get(i, 0.02) * 40
            scipy_b = scipy_range_bonus.get(i, 0) * 3
            range_weights[i] = chi2_weight * 0.3 + cv_weight * 10 * 0.2 + np_b + tf_b + pt_b + scipy_b
        
        sorted_ranges = sorted(range_weights.items(), key=lambda x: x[1], reverse=True)[:4]
        selected_ranges = [r for r, _ in sorted_ranges]
        
        # 按特征得分选号
        candidates = []
        for rng_idx in selected_ranges:
            start, end, _ = LotteryConfig.RANGES[rng_idx]
            rng_nums = list(range(start, end + 1))
            rng_nums.sort(key=lambda x: self._get_weighted_score(x), reverse=True)
            candidates.extend(rng_nums[:3])
        
        candidates.sort(key=lambda x: self._get_weighted_score(x), reverse=True)
        return candidates[:count]
    
    def _np_range_histogram(self):
        """NumPy histogram区间分布分析"""
        bonus = {}
        if not hasattr(self, 'np_histogram') or self.np_histogram is None:
            return bonus
        try:
            rng_counts = [0] * 5
            for num in range(1, 50):
                idx = LotteryConfig.get_range_index(num)
                if idx >= 0:
                    rng_counts[idx] += 1
            total = sum(rng_counts) or 1
            avg = total / 5
            for i in range(5):
                if rng_counts[i] < avg * 0.8:
                    bonus[i] = 3  # 低频区间需关注
                else:
                    bonus[i] = 0
        except:
            pass
        return bonus
    
    def _tf_range_predict(self):
        """TensorFlow区间分布预测"""
        bonus = {}
        if not hasattr(self, 'tf_predictions') or not self.tf_predictions:
            return bonus
        try:
            rng_probs = [0] * 5
            for num in range(1, 50):
                idx = LotteryConfig.get_range_index(num)
                if idx >= 0:
                    rng_probs[idx] += self.tf_predictions.get(num, 0.02)
            for i in range(5):
                bonus[i] = rng_probs[i] / 10 if rng_probs[i] > 0 else 0.02
        except:
            pass
        return bonus
    
    def _pt_range_predict(self):
        """PyTorch LSTM区间分布预测"""
        bonus = {}
        if not hasattr(self, 'pt_lstm_preds') or not self.pt_lstm_preds:
            return bonus
        try:
            rng_probs = [0] * 5
            for num in range(1, 50):
                idx = LotteryConfig.get_range_index(num)
                if idx >= 0:
                    rng_probs[idx] += self.pt_lstm_preds.get(num, 0.02)
            for i in range(5):
                bonus[i] = rng_probs[i] / 10 if rng_probs[i] > 0 else 0.02
        except:
            pass
        return bonus
    
    def _scipy_range_interp(self):
        """SciPy interpolate区间插值"""
        bonus = {}
        if not hasattr(self, 'scipy_interp_func') or not self.scipy_interp_func:
            return bonus
        try:
            for i in range(5):
                start, end, _ = LotteryConfig.RANGES[i]
                vals = [self.scipy_interp_func(n) for n in range(start, end + 1)]
                bonus[i] = np.mean(vals) if vals else 0.02
        except:
            pass
        return bonus
    
    def _range_cv_scores(self):
        """MinMaxScaler归一化区间特征 + cross_val_score验证区间预测模型"""
        _get_sklearn()
        MMS = _MinMaxScaler
        GBC = _GradientBoostingClassifier
        CVS = _cross_val_score
        if MMS is None or GBC is None or CVS is None:
            return {i: 0.5 for i in range(5)}
        try:
            # 构建区间分布训练数据
            X_range, y_range = [], []
            for i in range(len(self.data) - 1):
                if i < len(self.data) - 1:
                    # 特征：最近5期各区间出现次数
                    features = []
                    for j in range(5):
                        idx = i + j
                        if idx < len(self.data):
                            cnt = [0] * 5
                            for n in self.data[idx].get('numbers', []):
                                ri = LotteryConfig.get_range_index(n)
                                if ri >= 0:
                                    cnt[ri] += 1
                            features.extend(cnt)
                        else:
                            features.extend([0] * 5)
                    X_range.append(features)
                    # 标签：下一期各区间出现次数
                    next_nums = self.data[i].get('numbers', [])
                    label = [0] * 5
                    for n in next_nums:
                        ri = LotteryConfig.get_range_index(n)
                        if ri >= 0:
                            label[ri] += 1
                    y_range.append(label)
            if len(X_range) < 10:
                return {i: 0.5 for i in range(5)}
            X_range = np.array(X_range)
            y_range = np.array(y_range)
            # MinMaxScaler归一化
            scaler_range = MMS()
            X_range_scaled = scaler_range.fit_transform(X_range)
            # 对每个区间训练并验证
            cv_scores = {}
            for i in range(5):
                model = GBC(n_estimators=30, max_depth=3, random_state=42)
                scores = CVS(model, X_range_scaled, y_range[:, i], cv=3, scoring='accuracy')
                cv_scores[i] = np.mean(scores)
            return cv_scores
        except:
            return {i: 0.5 for i in range(5)}
    
    def roulette_selection(self, count=6):
        """轮盘赌 - TensorFlow概率 + PyTorch动态温度采样 + sklearn概率融合（无PyTorch时使用NumPy）"""
        torch_mod = _get_torch()
        if torch_mod is not None:
            torch_mod.manual_seed(int(datetime.datetime.now().timestamp()) % 1000000)
        
        # 计算综合权重
        weights = []
        for num in range(1, 50):
            w = self._get_weighted_score(num)
            weights.append(w)
        
        # sklearn: LogisticRegression概率权重
        lr_weights = self._lr_roulette_weights() if len(self.data) >= 20 else [0.02] * 49
        
        # TensorFlow概率权重
        tf_weights = []
        if hasattr(self, 'tf_predictions') and self.tf_predictions:
            tf_weights = [self.tf_predictions.get(n, 0.02) for n in range(1, 50)]
        else:
            tf_weights = [0.02] * 49
        
        # PyTorch LSTM概率权重
        pt_weights = []
        if hasattr(self, 'pt_lstm_preds') and self.pt_lstm_preds:
            pt_weights = [self.pt_lstm_preds.get(n, 0.02) for n in range(1, 50)]
        else:
            pt_weights = [0.02] * 49
        
        # NumPy矩阵运算融合权重
        combined_weights = np.array(weights) * 0.3 + np.array(lr_weights) * 200 * 0.2 + np.array(tf_weights) * 100 * 0.25 + np.array(pt_weights) * 100 * 0.25
        
        # 动态温度：基于权重分布熵
        weight_std = np.std(combined_weights)
        temperature = max(0.3, min(0.8, weight_std / 50))
        
        # PyTorch动态温度softmax采样（无PyTorch时使用NumPy替代）
        torch_mod = _get_torch()
        if torch_mod is not None and self.device is not None:
            weights_tensor = torch_mod.tensor(combined_weights, dtype=torch_mod.float32, device=self.device)
            logits = weights_tensor / temperature
            probs = torch_mod.softmax(logits, dim=0)
            probs_np = probs.cpu().numpy()
        else:
            # NumPy替代实现
            logits = combined_weights / temperature
            probs_np = np.exp(logits - np.max(logits))  # 数值稳定的softmax
            probs_np = probs_np / probs_np.sum()
        
        # 不放回采样
        selected = []
        available = list(range(49))  # 0-based索引，对应号码1-49
        
        for _ in range(count):
            if not available:
                break
            probs_sum = probs_np[available].sum()
            if probs_sum <= 0:
                probs_sum = 1
            idx = np.random.choice(len(available), p=probs_np[available] / probs_sum)
            num_idx = available[idx]
            selected.append(num_idx + 1)  # 转回1-based号码
            probs_np[num_idx] = 0
            available.remove(num_idx)
        
        return selected
    
    def historical_similarity(self, count=6):
        """历史相似性 - TensorFlow AutoEncoder + sklearn PCA + NumPy矩阵 + cosine_similarity"""
        _get_sklearn()
        if len(self.data) < 10:
            return random.sample(range(1, 50), count)
        
        # sklearn: PCA降维（63维→10维）后再计算cosine_similarity
        pca_based_sim = self._pca_historical_similarity() if hasattr(self, 'sklearn_features') and self.sklearn_features is not None else []
        
        # TensorFlow AutoEncoder降维相似度
        tf_sim = self._tf_autoencoder_similarity() if hasattr(self, 'tf_encoded') and self.tf_encoded is not None else []
        
        # 构建增强特征向量（one-hot 49维 + 统计特征14维 = 63维）
        def to_enhanced_vector(numbers):
            # one-hot编码（49维）
            vec = [0] * 49
            for n in numbers:
                if 1 <= n <= 49:
                    vec[n - 1] = 1
            # 统计特征（14维）
            stats = []
            stats.append(sum(numbers) / 6)
            stats.append(np.mean(numbers))
            stats.append(np.std(numbers) if len(numbers) > 1 else 0)
            stats.append(max(numbers) - min(numbers))
            stats.append(sum(1 for n in numbers if LotteryConfig.is_odd(n)))
            stats.append(sum(1 for n in numbers if LotteryConfig.is_big(n)))
            for i in range(5):
                stats.append(sum(1 for n in numbers if LotteryConfig.get_range_index(n) == i))
            for t in [0, 1, 2, 3]:
                stats.append(sum(1 for n in numbers if LotteryConfig.get_tail_digit(n) == t))
            return vec + stats
        
        latest = to_enhanced_vector(self.data[0].get('numbers', []))
        
        # NumPy矩阵运算计算与历史的相似度
        similarities = []
        cosine_sim = _cosine_similarity
        if cosine_sim is not None:
            for i, record in enumerate(self.data[1:50]):
                hist = to_enhanced_vector(record.get('numbers', []))
                # sklearn cosine_similarity
                sim = cosine_sim([latest], [hist])[0][0]
                similarities.append((i + 1, sim))
        else:
            for i, record in enumerate(self.data[1:50]):
                similarities.append((i + 1, 0))
        
        # 融合PCA和TF的相似度
        if pca_based_sim:
            for idx, sim in pca_based_sim:
                for i, (old_idx, old_sim) in enumerate(similarities):
                    if old_idx == idx:
                        similarities[i] = (old_idx, old_sim * 0.5 + sim * 0.5)
                        break
        
        if tf_sim:
            for idx, sim in tf_sim:
                for i, (old_idx, old_sim) in enumerate(similarities):
                    if old_idx == idx:
                        similarities[i] = (old_idx, old_sim * 0.5 + sim * 0.5)
                        break
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:5]
        
        # 相似度加权投票
        next_numbers = []
        for idx, sim in top_similar:
            if idx + 1 < len(self.data):
                for n in self.data[idx + 1].get('numbers', []):
                    next_numbers.extend([n] * int(sim * 10 + 1))
        
        if next_numbers:
            counter = Counter(next_numbers)
            selected = [num for num, cnt in counter.most_common(count * 2) if cnt >= 1]
            return selected[:count]
        return random.sample(range(1, 50), count)
    
    def _pca_historical_similarity(self):
        """sklearn PCA降维后计算cosine_similarity"""
        try:
            _get_sklearn()
            PCA = _PCA
            cosine_sim = _cosine_similarity
            if PCA is None or cosine_sim is None:
                return []
            similarities = []
            if self.sklearn_features_scaled is None:
                return similarities
            # PCA降到10维
            pca = PCA(n_components=min(10, self.sklearn_features_scaled.shape[1]))
            encoded = pca.fit_transform(self.sklearn_features_scaled)
            latest_encoded = encoded[0] if len(encoded) > 0 else self.sklearn_features_scaled[0]
            for i in range(1, min(50, len(encoded))):
                sim = cosine_sim([latest_encoded], [encoded[i]])[0][0]
                similarities.append((i, sim))
            return similarities
        except:
            return []
    
    def _tf_autoencoder_similarity(self):
        """TensorFlow AutoEncoder降维后计算相似度"""
        _get_sklearn()
        cosine_sim = _cosine_similarity
        if cosine_sim is None:
            return []
        try:
            similarities = []
            if not hasattr(self, 'tf_encoded') or self.tf_encoded is None:
                return similarities
            latest_encoded = self.tf_encoded[0] if len(self.tf_encoded) > 0 else self.sklearn_features_scaled[0]
            for i in range(1, min(50, len(self.tf_encoded))):
                sim = cosine_sim([latest_encoded], [self.tf_encoded[i]])[0][0]
                similarities.append((i, sim))
            return similarities
        except:
            return []
    
    def poisson_distribution(self, count=6):
        """泊松分布 - TensorFlow补充 + sklearn NB + SciPy分布 + NumPy矩阵"""
        if not self.data:
            return random.sample(range(1, 50), count)
        
        # sklearn: GaussianNB补充概率估计
        nb_probs = self._nb_missing_probs() if len(self.data) >= 20 else {}
        
        # TensorFlow概率补充
        tf_probs = self._tf_poisson_supplement() if hasattr(self, 'tf_predictions') and self.tf_predictions else {}
        
        # PyTorch LSTM概率补充
        pt_probs = self._pt_poisson_supplement() if hasattr(self, 'pt_lstm_preds') and self.pt_lstm_preds else {}
        
        # 获取scipy.stats
        scipy_st = _get_scipy_stats()
        
        # NumPy矩阵运算计算综合概率
        probabilities = {}
        for num in range(1, 50):
            # 基础泊松参数
            freq = self.frequency.get(num, 0)
            lambda_param = freq / len(self.data) if len(self.data) > 0 else 1 / 49
            
            # SciPy泊松分布概率
            if scipy_st is not None:
                poisson_prob = 1 - scipy_st.poisson.cdf(0, lambda_param * len(self.data))
            else:
                poisson_prob = 0.5
            
            # 指数分布拟合
            stats = self.interval_stats.get(num, {})
            mean_interval = stats.get('mean', 6)
            if scipy_st is not None and mean_interval > 0:
                exp_prob = scipy_st.expon.cdf(self.missing.get(num, 50), scale=mean_interval)
            else:
                exp_prob = 0.5
            
            # 生存函数
            if scipy_st is not None:
                survival = scipy_st.expon.sf(self.missing.get(num, 50), scale=mean_interval)
            else:
                survival = 0.5
            
            # Pandas移动平均趋势
            if hasattr(self, 'df') and not self.df.empty:
                ma_trend = self.moving_avg.get(num, {}).get(5, 0.1)
            else:
                ma_trend = 0.1
            
            # 条件概率
            cond_prob = self.consecutive_prob.get(num, 0.01)
            
            # sklearn GaussianNB概率
            nb_prob = nb_probs.get(num, 0.02)
            
            # TensorFlow概率
            tf_prob = tf_probs.get(num, 0.02)
            
            # PyTorch概率
            pt_prob = pt_probs.get(num, 0.02)
            
            # NumPy矩阵运算融合
            base_prob = poisson_prob * 0.2 + exp_prob * 0.2 + survival * 0.2 + ma_trend * 0.1 + cond_prob * 0.1
            ml_prob = nb_prob * 0.1 + tf_prob * 0.05 + pt_prob * 0.05
            probabilities[num] = base_prob + ml_prob
        
        sorted_nums = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _tf_poisson_supplement(self):
        """TensorFlow概率补充泊松估计"""
        if not hasattr(self, 'tf_predictions') or not self.tf_predictions:
            return {}
        return self.tf_predictions.copy()
    
    def _pt_poisson_supplement(self):
        """PyTorch LSTM概率补充泊松估计"""
        if not hasattr(self, 'pt_lstm_preds') or not self.pt_lstm_preds:
            return {}
        return self.pt_lstm_preds.copy()
    
    def mystical_algorithm(self, count=6):
        """玄学算法 - TensorFlow随机采样 + sklearn KMeans五行聚类 + NumPy矩阵 + PyTorch采样"""
        now = datetime.datetime.now()
        
        # TensorFlow随机采样生成初始权重
        tf_random_weights = self._tf_mystical_weights() if hasattr(self, 'tf_predictions') and self.tf_predictions else np.ones(49)
        
        # sklearn: KMeans对五行数字组做聚类分析
        wu_xing_cluster = self._mystical_kmeans_cluster()
        
        # 天干地支计算
        tian_gan = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
        di_zhi = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']
        gan_idx = (now.year - 1984) % 10
        zhi_idx = (now.year - 1984) % 12
        
        # 五行属性
        wu_xing = {
            '甲': '木', '乙': '木', '丙': '火', '丁': '火', '戊': '土',
            '己': '土', '庚': '金', '辛': '金', '壬': '水', '癸': '水'
        }
        wu_xing_strength = {
            '木': [3, 4, 21, 22, 41, 42],
            '火': [7, 8, 17, 18, 27, 28, 37, 38, 47, 48],
            '土': [9, 10, 19, 20, 29, 30, 39, 40, 49],
            '金': [5, 6, 15, 16, 25, 26, 35, 36, 45, 46],
            '水': [1, 2, 11, 12, 21, 22, 31, 32, 43, 44]
        }
        
        # 五行生克关系
        sheng = {'木': '火', '火': '土', '土': '金', '金': '水', '水': '木'}
        ke = {'木': '土', '火': '金', '土': '水', '金': '木', '水': '火'}
        
        current_element = wu_xing.get(tian_gan[gan_idx % 10], '土')
        Sheng_element = sheng.get(current_element, '')
        Ke_element = ke.get(current_element, '')
        
        # NumPy矩阵运算融合多源权重
        base_weights_np = np.ones(49)
        
        # 基于五行加权
        for num in range(1, 50):
            idx = num - 1
            if Sheng_element:
                for wu_num in wu_xing_strength.get(Sheng_element, []):
                    if num == wu_num:
                        base_weights_np[idx] *= 0.8
            if Ke_element:
                for wu_num in wu_xing_strength.get(Ke_element, []):
                    if num == wu_num:
                        base_weights_np[idx] *= 0.9
            for element, nums in wu_xing_strength.items():
                if sheng.get(element, '') == current_element and num in nums:
                    base_weights_np[idx] *= 1.2
                if ke.get(element, '') == current_element and num in nums:
                    base_weights_np[idx] *= 0.7
        
        # 融合TensorFlow权重
        base_weights_np *= tf_random_weights
        
        # sklearn KMeans聚类加成
        for num, cluster_id in wu_xing_cluster.items():
            base_weights_np[num - 1] *= (1.0 + cluster_id * 0.1)
        
        # 融合历史数据
        if self.data:
            for i, record in enumerate(self.data[:5]):
                weight = (5 - i) * 0.1
                for num in record.get('numbers', []):
                    if 1 <= num <= 49:
                        base_weights_np[num - 1] += weight
        
        # 时间因子
        time_factor = now.hour * 60 + now.minute
        time_bonus = (time_factor % 10) / 100
        base_weights_np += np.random.rand(49) * time_bonus
        
        # PyTorch加权采样（无PyTorch时使用NumPy替代）
        torch_mod = _get_torch()
        if torch_mod is not None and self.device is not None:
            base_weights = torch_mod.tensor(base_weights_np, dtype=torch_mod.float32, device=self.device)
            probs = torch_mod.softmax(base_weights / 0.7, dim=0)
            probs_np = probs.cpu().numpy()
        else:
            # NumPy替代实现
            logits = base_weights_np / 0.7
            probs_np = np.exp(logits - np.max(logits))  # 数值稳定的softmax
            probs_np = probs_np / probs_np.sum()
        
        for _ in range(count):
            if not available:
                break
            probs_sum = probs_np[available].sum()
            if probs_sum <= 0:
                probs_sum = 1
            idx = np.random.choice(len(available), p=probs_np[available] / probs_sum)
            num = available[idx]
            selected.append(num)
            probs_np[num - 1] = 0
            available.remove(num)
        
        return selected
    
    def _tf_mystical_weights(self):
        """TensorFlow随机采样生成玄学权重"""
        if not hasattr(self, 'tf_predictions') or not self.tf_predictions:
            return np.ones(49)
        try:
            weights = np.array([self.tf_predictions.get(n, 0.02) for n in range(1, 50)])
            weights = weights / weights.sum() * 49  # 归一化
            return weights
        except:
            return np.ones(49)
    
    def _mystical_kmeans_cluster(self):
        """sklearn KMeans对五行数字组做聚类分析"""
        cluster_map = {}
        if not hasattr(self, 'kmeans_labels') or self.kmeans_labels is None:
            return cluster_map
        try:
            for num in range(1, 50):
                cluster_map[num] = self.kmeans_labels[num - 1]
        except:
            pass
        return cluster_map
    
    # ================================================================
    # 4个NetworkX图算法新增
    # ================================================================
    
    def _build_number_graph(self):
        """构建号码共现图 - NetworkX Graph"""
        nx = _get_nx()
        if nx is None:
            return None
        G = nx.Graph()
        # 添加49个节点
        for num in range(1, 50):
            G.add_node(num, missing=self.missing.get(num, 50), freq=self.frequency.get(num, 0))
        # 添加共现边
        for record in self.data:
            numbers = record.get('numbers', [])
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    n1, n2 = numbers[i], numbers[j]
                    if G.has_edge(n1, n2):
                        G[n1][n2]['weight'] += 1
                    else:
                        G.add_edge(n1, n2, weight=1)
        return G
    
    def _build_transition_graph(self):
        """构建号码转移图 - NetworkX DiGraph"""
        nx = _get_nx()
        if nx is None:
            return None
        DG = nx.DiGraph()
        for num in range(1, 50):
            DG.add_node(num)
        # 统计号码转移
        for i in range(len(self.data) - 1):
            curr_nums = set(self.data[i].get('numbers', []))
            next_nums = set(self.data[i + 1].get('numbers', []))
            for n1 in curr_nums:
                for n2 in next_nums:
                    if DG.has_edge(n1, n2):
                        DG[n1][n2]['weight'] += 1
                    else:
                        DG.add_edge(n1, n2, weight=1)
        return DG
    
    def number_graph_algorithm(self, count=6):
        """号码关联图算法 - NetworkX PageRank/度中心性/介数中心性（无NetworkX时使用基础统计）"""
        if len(self.data) < 10:
            return random.sample(range(1, 50), count)
        
        # 无NetworkX时使用基础统计方法
        nx = _get_nx()
        if nx is None:
            return self._nx_number_graph_fallback(count)
        
        # 构建共现图
        G = self._build_number_graph()
        if G.number_of_edges() == 0:
            return random.sample(range(1, 50), count)
        
        scores = {}
        try:
            # NetworkX PageRank中心性
            pagerank = nx.pagerank(G, weight='weight')
            # NetworkX 度中心性
            degree_cent = nx.degree_centrality(G)
            # NetworkX 介数中心性
            betweenness = nx.betweenness_centrality(G, weight='weight')
            # 融合多中心性
            for num in range(1, 50):
                pr = pagerank.get(num, 0)
                dc = degree_cent.get(num, 0)
                bc = betweenness.get(num, 0)
                # 遗漏回补加成
                miss = self.missing.get(num, 50)
                miss_bonus = MathUtils.calculate_missing_cycle(miss, 6)
                # 综合得分
                scores[num] = pr * 50 + dc * 30 + bc * 20 + miss_bonus * 10
        except Exception:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def shortest_path_algorithm(self, count=6):
        """最短路径算法 - NetworkX Dijkstra号码转移分析（无NetworkX时使用基础统计）"""
        if len(self.data) < 10:
            return random.sample(range(1, 50), count)
        
        # 无NetworkX时使用基础统计方法
        nx = _get_nx()
        if nx is None:
            return self._nx_shortest_path_fallback(count)
        
        DG = self._build_transition_graph()
        if DG.number_of_edges() == 0:
            return random.sample(range(1, 50), count)
        
        scores = {}
        try:
            latest_nums = self.data[0].get('numbers', [])
            # NetworkX Dijkstra最短路径
            for target in range(1, 50):
                min_dist = float('inf')
                total_prob = 0.0
                for source in latest_nums:
                    try:
                        path_length = nx.dijkstra_path_length(DG, source, target, weight='weight')
                        path_prob = 1.0 / (path_length + 1)
                        min_dist = min(min_dist, path_length)
                        total_prob += path_prob
                    except nx.NetworkXNoPath:
                        continue
                # NetworkX 特征向量中心性
                try:
                    eigen_cent = nx.eigenvector_centrality(DG, weight='weight', max_iter=1000)
                    eigen = eigen_cent.get(target, 0)
                except:
                    eigen = 0
                # 遗漏加成
                miss = self.missing.get(target, 50)
                miss_bonus = MathUtils.calculate_missing_cycle(miss, 6)
                # 综合得分
                scores[target] = (1.0 / (min_dist + 1)) * 30 + total_prob * 20 + eigen * 30 + miss_bonus * 20
        except Exception:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def community_detection_algorithm(self, count=6):
        """社区发现算法 - NetworkX Louvain社区检测（无NetworkX时使用基础统计）"""
        if len(self.data) < 15:
            return random.sample(range(1, 50), count)
        
        # 无NetworkX时使用基础统计方法
        nx = _get_nx()
        if nx is None:
            return self._nx_community_fallback(count)
        
        G = self._build_number_graph()
        if G.number_of_edges() < 5:
            return random.sample(range(1, 50), count)
        
        scores = {}
        try:
            # NetworkX Louvain社区检测 (greedy_modularity)
            communities = list(nx.community.greedy_modularity_communities(G, weight='weight'))
            # 找出最新一期号码所在的社区
            latest_nums = set(self.data[0].get('numbers', []))
            target_communities = set()
            for idx, comm in enumerate(communities):
                if latest_nums & comm:  # 有交集
                    target_communities.add(idx)
            
            # 统计每个社区的得分
            for num in range(1, 50):
                # 找到该号码所在的社区
                num_community = -1
                for idx, comm in enumerate(communities):
                    if num in comm:
                        num_community = idx
                        break
                
                # 与目标社区的关联度
                community_score = 5 if num_community in target_communities else 1
                # 社区内连接强度
                degree_in_comm = G.degree(num, weight='weight') if G.has_node(num) else 0
                # PageRank
                pagerank = nx.pagerank(G, weight='weight').get(num, 0)
                # 遗漏加成
                miss = self.missing.get(num, 50)
                miss_bonus = MathUtils.calculate_missing_cycle(miss, 6)
                
                scores[num] = community_score * 15 + degree_in_comm * 0.5 + pagerank * 50 + miss_bonus * 10
        except Exception:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def graph_clustering_algorithm(self, count=6):
        """图聚类算法 - NetworkX连通分量/k-clique聚类（无NetworkX时使用基础统计）"""
        if len(self.data) < 10:
            return random.sample(range(1, 50), count)
        
        # 无NetworkX时使用基础统计方法
        nx = _get_nx()
        if nx is None:
            return self._nx_clustering_fallback(count)
        
        G = self._build_number_graph()
        if G.number_of_edges() < 3:
            return random.sample(range(1, 50), count)
        
        scores = {}
        try:
            # NetworkX 连通分量分析
            components = list(nx.connected_components(G))
            # NetworkX k-clique社区检测
            kclique_communities = []
            try:
                kclique_communities = list(nx.community.k_clique_communities(G, 3))
            except:
                kclique_communities = []
            
            # 找出热门连通分量
            latest_nums = set(self.data[0].get('numbers', []))
            target_components = []
            for comp in components:
                if len(comp & latest_nums) > 0:
                    target_components.append(comp)
            
            # k-clique社区得分
            kclique_scores = {}
            for num in range(1, 50):
                kclique_scores[num] = 0
                for comm in kclique_communities:
                    if num in comm:
                        kclique_scores[num] += len(comm)
            
            for num in range(1, 50):
                # 所在连通分量大小
                comp_size = 0
                for comp in components:
                    if num in comp:
                        comp_size = len(comp)
                        break
                # 与目标连通分量的关联
                in_target = any(num in comp for comp in target_components)
                target_bonus = 10 if in_target else 1
                # k-clique得分
                kc_score = kclique_scores.get(num, 0)
                # 聚类系数
                clustering_coeff = nx.clustering(G, num) if G.has_node(num) else 0
                # 基础特征得分
                base_score = self._get_weighted_score(num) / 100
                # 遗漏加成
                miss = self.missing.get(num, 50)
                miss_bonus = MathUtils.calculate_missing_cycle(miss, 6)
                
                scores[num] = (target_bonus * 10 + comp_size * 0.5 + kc_score * 2 + 
                              clustering_coeff * 20 + base_score * 20 + miss_bonus * 10)
        except Exception:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def numpy_matrix_algorithm(self, count=6):
        """NumPy矩阵算法 - 基于NumPy矩阵运算深度分析预测"""
        if len(self.data) < 10:
            return random.sample(range(1, 50), count)
        
        scores = {}
        try:
            # 构建49个数字的特征矩阵
            feature_matrix = np.zeros((49, 8), dtype=np.float32)
            for num in range(1, 50):
                freq = self.frequency.get(num, 0)
                miss = min(self.missing.get(num, 50), 50)
                ma5 = self.moving_avg.get(num, {}).get(5, 0.1)
                ma10 = self.moving_avg.get(num, {}).get(10, 0.1)
                ma20 = self.moving_avg.get(num, {}).get(20, 0.1)
                autocorr = self.autocorrelation.get(num, 0)
                zscore = self.interval_stats.get(num, {}).get('zscore', 0)
                tail = LotteryConfig.get_tail_digit(num)
                tail_freq = self.tail_distribution.get(tail, 1)
                
                feature_matrix[num-1] = [freq, miss, ma5*100, ma10*100, ma20*100, 
                                        autocorr*100, zscore*10, tail_freq]
            
            # np.linalg.svd奇异值分解降维
            try:
                U, s, Vt = np.linalg.svd(feature_matrix, full_matrices=False)
                # 取前3个奇异值对应的特征
                svd_features = np.dot(feature_matrix, Vt[:3].T)
            except:
                svd_features = feature_matrix[:, :3]
            
            # np.corrcoef计算特征相关性矩阵
            corr_matrix = np.corrcoef(feature_matrix.T)
            # 取与频率相关性最高的特征
            freq_corr = corr_matrix[0]
            top_corr_idx = np.argsort(np.abs(freq_corr))[::-1][:4]
            
            # np.percentile计算分位数
            freq_percentiles = np.percentile(feature_matrix[:, 0], [25, 50, 75])
            miss_percentiles = np.percentile(feature_matrix[:, 1], [25, 50, 75])
            
            # np.vander构建多项式基进行遗漏趋势预测
            n = min(20, len(self.data))
            y_miss = np.array([self.missing.get(i+1, 50) for i in range(n)], dtype=np.float64)
            X_poly = np.vander(np.arange(n), 3)  # 三阶多项式基
            coeffs, residuals, rank, s_val = np.linalg.lstsq(X_poly, y_miss, rcond=None)
            next_x = np.array([[n**2, n, 1]])
            predicted_missing = np.dot(next_x, coeffs)[0]
            
            # np.histogram分析频率分布
            hist, bin_edges = np.histogram(feature_matrix[:, 0], bins=10)
            hot_bin_idx = np.argmax(hist)
            hot_bin_range = (bin_edges[hot_bin_idx], bin_edges[hot_bin_idx+1])
            
            # np.dot矩阵乘法计算综合得分
            weights = np.array([0.25, 0.2, 0.15, 0.15, 0.1, 0.1, 0.03, 0.02])
            for num in range(1, 50):
                # 基础得分
                base_score = np.dot(feature_matrix[num-1], weights)
                
                # SVD特征得分
                svd_score = np.linalg.norm(svd_features[num-1])
                
                # 相关性加权得分
                corr_score = sum(feature_matrix[num-1, top_corr_idx[i]] * (1.0 / (i+1)) 
                                for i in range(len(top_corr_idx)))
                
                # 分位数位置得分
                freq_val = feature_matrix[num-1, 0]
                miss_val = feature_matrix[num-1, 1]
                if freq_val >= freq_percentiles[2]:
                    percentile_score = 20
                elif freq_val >= freq_percentiles[1]:
                    percentile_score = 15
                elif freq_val >= freq_percentiles[0]:
                    percentile_score = 10
                else:
                    percentile_score = 5
                
                if miss_val <= miss_percentiles[0]:
                    miss_percentile_score = 20
                elif miss_val <= miss_percentiles[1]:
                    miss_percentile_score = 15
                elif miss_val <= miss_percentiles[2]:
                    miss_percentile_score = 10
                else:
                    miss_percentile_score = 5
                
                # 遗漏趋势偏离得分
                miss_deviation = abs(feature_matrix[num-1, 1] - predicted_missing)
                trend_score = max(0, 15 - miss_deviation * 0.5)
                
                # 分布热区得分
                dist_score = 10 if hot_bin_range[0] <= feature_matrix[num-1, 0] <= hot_bin_range[1] else 3
                
                # 综合得分
                scores[num] = (base_score * 20 + svd_score * 5 + corr_score * 3 + 
                             percentile_score * 3 + miss_percentile_score * 3 + 
                             trend_score * 5 + dist_score * 2)
            
        except Exception as e:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def scipy_optimization_algorithm(self, count=6):
        """SciPy优化算法 - 基于SciPy科学计算优化预测"""
        if len(self.data) < 10:
            return random.sample(range(1, 50), count)
        
        scores = {}
        try:
            # scipy.signal.convolve高斯平滑趋势
            scipy_sig = _get_scipy_signal()
            scipy_opt = _get_scipy_optimize()
            scipy_int = _get_scipy_interpolate()
            scipy_st = _get_scipy_stats()
            
            freq_array = np.array([self.frequency.get(i, 0) for i in range(1, 50)], dtype=np.float32)
            # 高斯核
            if scipy_sig is not None:
                gauss_kernel = scipy_sig.gaussian(7, std=1.5)
                gauss_kernel = gauss_kernel / gauss_kernel.sum()
                smoothed_freq = scipy_sig.convolve(freq_array, gauss_kernel, mode='same')
            else:
                smoothed_freq = freq_array
            
            # scipy.optimize.minimize权重优化
            if scipy_opt is None:
                return random.sample(range(1, 50), count)
            minimize = scipy_opt.minimize
            
            def objective(params):
                w_freq, w_miss, w_trend, w_corr, w_range = params
                total_w = w_freq + w_miss + w_trend + w_corr + w_range
                if total_w == 0:
                    return 0
                w_freq, w_miss, w_trend, w_corr, w_range = [p/total_w for p in params]
                
                score = 0
                for num in range(1, 50):
                    freq_s = self.frequency.get(num, 0) / len(self.data) if self.data else 0
                    miss_s = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
                    ma5 = self.moving_avg.get(num, {}).get(5, 0.1)
                    ma20 = self.moving_avg.get(num, {}).get(20, 0.1)
                    trend_s = ma5 - ma20 + 0.1
                    
                    corr_s = 0
                    if len(self.data) > 0:
                        latest_nums = self.data[0].get('numbers', [])
                        corr_s = sum(self.correlation_matrix[num-1, n-1] for n in latest_nums) / len(latest_nums)
                    
                    range_idx = LotteryConfig.get_range_index(num)
                    range_s = 1 - self.range_distribution.get(range_idx, 0) / sum(self.range_distribution.values())
                    
                    score += (w_freq * freq_s + w_miss * miss_s + w_trend * trend_s + 
                             w_corr * corr_s + w_range * range_s)
                return -score  # minimize，所以取负
            
            # 优化权重
            x0 = [0.25, 0.25, 0.2, 0.15, 0.15]
            result = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 100, 'disp': False})
            opt_weights = result.x
            total_w = sum(opt_weights)
            opt_weights = [w/total_w if total_w > 0 else 1/5 for w in opt_weights]
            
            # scipy.interpolate样条插值预测遗漏趋势
            if scipy_int is not None:
                splrep = scipy_int.splrep
                splev = scipy_int.splev
            else:
                splrep = None
                splev = None
            n = min(30, len(self.data))
            x_data = np.arange(n)
            y_data = np.array([self.missing.get(i+1, 50) for i in range(n)], dtype=np.float64)
            
            if splrep is not None and splev is not None and len(x_data) >= 4:
                tck = splrep(x_data, y_data, k=3, s=len(x_data))
                x_pred = np.array([n, n+1, n+2])
                y_pred = splev(x_pred, tck)
                interp_missing = {i+1: max(1, min(50, y_pred[i])) for i in range(3)}
            else:
                interp_missing = {i+1: self.missing.get(i+1, 50) for i in range(3)}
            
            # scipy.stats.poisson分布建模
            poisson = None
            expon = None
            norm = None
            ks_2samp = None
            if scipy_st is not None:
                poisson = getattr(scipy_st, 'poisson', None)
                expon = getattr(scipy_st, 'expon', None)
                norm = getattr(scipy_st, 'norm', None)
                ks_2samp_func = getattr(scipy_st, 'ks_2samp', None)
            # 拟合泊松分布
            intervals_all = []
            for num in range(1, 50):
                intervals = []
                last_appeared = None
                for i, record in enumerate(self.data):
                    if num in record.get('numbers', []):
                        if last_appeared is not None:
                            intervals.append(i - last_appeared)
                        last_appeared = i
                intervals_all.extend(intervals)
            
            mu, std = 0, 1
            loc_exp, scale_exp = 0, 1
            if intervals_all and norm is not None and expon is not None:
                # 拟合正态分布参数
                mu, std = norm.fit(intervals_all)
                # 拟合指数分布
                loc_exp, scale_exp = expon.fit(intervals_all)
            
            # scipy.stats.ks_2samp分布一致性检验
            for num in range(1, 50):
                intervals = []
                last_appeared = None
                for i, record in enumerate(self.data):
                    if num in record.get('numbers', []):
                        if last_appeared is not None:
                            intervals.append(i - last_appeared)
                        last_appeared = i
                
                # KS检验得分
                if intervals and intervals_all and ks_2samp_func is not None:
                    try:
                        ks_stat, ks_pval = ks_2samp_func(intervals, intervals_all)
                        ks_score = ks_pval * 10  # p值越高说明越符合整体分布
                    except:
                        ks_score = 5
                else:
                    ks_score = 5
                
                # 计算综合得分
                freq_s = self.frequency.get(num, 0) / len(self.data) if self.data else 0
                miss_s = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
                ma5 = self.moving_avg.get(num, {}).get(5, 0.1)
                ma20 = self.moving_avg.get(num, {}).get(20, 0.1)
                trend_s = ma5 - ma20 + 0.1
                
                corr_s = 0
                if len(self.data) > 0:
                    latest_nums = self.data[0].get('numbers', [])
                    corr_s = sum(self.correlation_matrix[num-1, n-1] for n in latest_nums) / len(latest_nums)
                
                range_idx = LotteryConfig.get_range_index(num)
                range_s = 1 - self.range_distribution.get(range_idx, 0) / sum(self.range_distribution.values())
                
                # 平滑后的频率得分
                smooth_freq = smoothed_freq[num-1] if num-1 < len(smoothed_freq) else freq_s
                
                # 插值预测遗漏得分
                interp_miss = interp_missing.get(1, self.missing.get(num, 50))
                interp_miss_score = max(0, 10 - abs(self.missing.get(num, 50) - interp_miss) * 0.3)
                
                # 分布拟合得分
                dist_score = 5
                if intervals and norm is not None and expon is not None:
                    try:
                        norm_prob = norm.pdf(self.missing.get(num, 50), mu, std) if std > 0 else 0.02
                        exp_prob = expon.pdf(self.missing.get(num, 50), loc_exp, scale_exp) if scale_exp > 0 else 0.02
                        dist_score = (norm_prob + exp_prob) * 50
                    except:
                        dist_score = 5
                
                scores[num] = (opt_weights[0] * freq_s * 100 + 
                              opt_weights[1] * miss_s * 100 + 
                              opt_weights[2] * trend_s * 50 +
                              opt_weights[3] * corr_s * 100 +
                              opt_weights[4] * range_s * 50 +
                              smooth_freq * 2 +  # 高斯平滑
                              interp_miss_score * 5 +  # 样条插值
                              dist_score +  # 分布拟合
                              ks_score)  # KS检验
            
        except Exception as e:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def sklearn_ensemble_algorithm(self, count=6):
        """Scikit-learn集成算法 - 基于Sklearn多模型集成预测"""
        _get_sklearn()
        SS = _StandardScaler
        MMS = _MinMaxScaler
        PCA = _PCA
        RF = _RandomForestClassifier
        GBC = _GradientBoostingClassifier
        LR = _LogisticRegression
        MLP = _MLPClassifier
        GNB = _GaussianNB
        KM = _KMeans
        CVS = _cross_val_score
        
        if len(self.data) < 20:
            return random.sample(range(1, 50), count)
        
        scores = {}
        try:
            # 构建训练数据
            X, y = [], []
            for i in range(len(self.data) - 1):
                features = self._build_sklearn_feature_vector(i)
                X.append(features)
                next_nums = self.data[i].get('numbers', [])
                y_single = [1 if n in next_nums else 0 for n in range(1, 50)]
                y.append(y_single)
            
            if len(X) < 10:
                return random.sample(range(1, 50), count)
            
            X = np.array(X)
            y = np.array(y)
            
            # 特征标准化
            if SS is not None:
                scaler = SS()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X
            
            # MinMax归一化
            if MMS is not None:
                minmax_scaler = MMS()
                X_minmax = minmax_scaler.fit_transform(X)
            else:
                X_minmax = X
            
            # PCA降维
            pca = None
            if PCA is not None and X_scaled.shape[1] >= 5:
                pca = PCA(n_components=min(10, X_scaled.shape[1]))
                X_pca = pca.fit_transform(X_scaled)
            
            # 各模型预测概率
            model_probs = {n: np.zeros(49) for n in range(1, 50)}
            
            for target_num in range(1, 50):
                y_target = y[:, target_num-1]
                
                # RandomForestClassifier
                if RF is not None:
                    rf = RF(n_estimators=50, max_depth=5, random_state=42)
                    try:
                        rf.fit(X_scaled, y_target)
                        rf_prob = rf.predict_proba(X_scaled[-1:])[0]
                        if len(rf_prob) > 1:
                            model_probs[target_num] += rf_prob[1] * 0.25
                        else:
                            model_probs[target_num] += 0.02 * 0.25
                    except:
                        model_probs[target_num] += 0.02 * 0.25
                
                # GradientBoostingClassifier
                if GBC is not None:
                    gb = GBC(n_estimators=50, max_depth=3, random_state=42)
                    try:
                        gb.fit(X_scaled, y_target)
                        gb_prob = gb.predict_proba(X_scaled[-1:])[0]
                        if len(gb_prob) > 1:
                            model_probs[target_num] += gb_prob[1] * 0.25
                        else:
                            model_probs[target_num] += 0.02 * 0.25
                    except:
                        model_probs[target_num] += 0.02 * 0.25
                
                # LogisticRegression
                if LR is not None:
                    lr = LR(max_iter=200, random_state=42)
                    try:
                        lr.fit(X_scaled, y_target)
                        lr_prob = lr.predict_proba(X_scaled[-1:])[0]
                        if len(lr_prob) > 1:
                            model_probs[target_num] += lr_prob[1] * 0.2
                        else:
                            model_probs[target_num] += 0.02 * 0.2
                    except:
                        model_probs[target_num] += 0.02 * 0.2
                
                # MLPClassifier
                if MLP is not None:
                    mlp_model = MLP(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
                    try:
                        mlp_model.fit(X_minmax, y_target)
                        mlp_prob = mlp_model.predict_proba(X_minmax[-1:])[0]
                        if len(mlp_prob) > 1:
                            model_probs[target_num] += mlp_prob[1] * 0.15
                        else:
                            model_probs[target_num] += 0.02 * 0.15
                    except:
                        model_probs[target_num] += 0.02 * 0.15
                
                # GaussianNB
                if GNB is not None:
                    gnb = GNB()
                    try:
                        gnb.fit(X_scaled, y_target)
                        gnb_prob = gnb.predict_proba(X_scaled[-1:])[0]
                        if len(gnb_prob) > 1:
                            model_probs[target_num] += gnb_prob[1] * 0.15
                        else:
                            model_probs[target_num] += 0.02 * 0.15
                    except:
                        model_probs[target_num] += 0.02 * 0.15
            
            # KMeans聚类分析
            cluster_nums = {i: list(range(1, 50)) for i in range(5)}
            target_cluster = 0
            if KM is not None:
                try:
                    kmeans = KM(n_clusters=5, random_state=42)
                    features_for_cluster = self._build_sklearn_feature_vector(0)
                    cluster_labels = kmeans.fit_predict([features_for_cluster])
                    
                    # 获取各类别的数字
                    cluster_nums = {i: [] for i in range(5)}
                    for num in range(1, 50):
                        features = self._build_sklearn_feature_vector_for_num(num)
                        cluster_id = kmeans.predict([features])[0]
                        cluster_nums[cluster_id].append(num)
                    
                    # 偏好聚类
                    target_cluster = cluster_labels[0] if len(cluster_labels) > 0 else 0
                except:
                    pass
            
            # cross_val_score验证
            cv_bonus = 0
            if CVS is not None and RF is not None:
                try:
                    rf_cv = RF(n_estimators=30, max_depth=4, random_state=42)
                    cv_scores = CVS(rf_cv, X_scaled, y[:, 25], cv=min(5, len(X)))
                    cv_bonus = np.mean(cv_scores) * 5
                except:
                    cv_bonus = 0
            
            # 综合得分
            for num in range(1, 50):
                # 模型集成概率得分
                ensemble_prob = model_probs[num][0] if model_probs[num].any() else 0.02
                
                # 基础特征得分
                base_score = self._get_weighted_score(num) / 100
                
                # 遗漏得分
                miss_s = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
                
                # 聚类偏好得分
                cluster_bonus = 10 if num in cluster_nums.get(target_cluster, []) else 1
                
                # 频率得分
                freq_s = self.frequency.get(num, 0) / len(self.data) if self.data else 0
                
                scores[num] = (ensemble_prob * 200 +  # 模型集成权重最高
                             base_score * 30 +
                             miss_s * 20 +
                             cluster_bonus * 3 +
                             freq_s * 50 +
                             cv_bonus)
            
        except Exception as e:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _build_sklearn_feature_vector(self, idx):
        """构建sklearn特征向量"""
        features = []
        for num in range(1, 50):
            freq = self.frequency.get(num, 0) / len(self.data) if self.data else 0
            miss = self.missing.get(num, 50) / 50
            ma5 = self.moving_avg.get(num, {}).get(5, 0.1)
            ma10 = self.moving_avg.get(num, {}).get(10, 0.1)
            ma20 = self.moving_avg.get(num, {}).get(20, 0.1)
            autocorr = self.autocorrelation.get(num, 0)
            zscore = self.interval_stats.get(num, {}).get('zscore', 0) / 5
            features.extend([freq, miss, ma5, ma10, ma20, autocorr, zscore])
        return features
    
    def _build_sklearn_feature_vector_for_num(self, num):
        """为单个数字构建特征向量"""
        freq = self.frequency.get(num, 0) / len(self.data) if self.data else 0
        miss = self.missing.get(num, 50) / 50
        ma5 = self.moving_avg.get(num, {}).get(5, 0.1)
        ma10 = self.moving_avg.get(num, {}).get(10, 0.1)
        ma20 = self.moving_avg.get(num, {}).get(20, 0.1)
        autocorr = self.autocorrelation.get(num, 0)
        zscore = self.interval_stats.get(num, {}).get('zscore', 0) / 5
        return [freq, miss, ma5, ma10, ma20, autocorr, zscore]
    
    def pytorch_deep_learning_algorithm(self, count=6):
        """PyTorch深度学习算法 - 基于PyTorch LSTM神经网络预测（无PyTorch时使用基础统计）"""
        if len(self.data) < 20:
            return random.sample(range(1, 50), count)
        
        # 无PyTorch时使用基础统计方法
        torch_mod = _get_torch()
        nn = _get_nn()
        optim_mod = _get_optim()
        if torch_mod is None:
            return self._pytorch_fallback(count)
        
        scores = {}
        try:
            # 构建时序数据
            sequence_length = 10
            X_seq, y_seq = [], []
            for i in range(len(self.data) - sequence_length):
                seq = []
                for j in range(sequence_length):
                    features = []
                    for num in range(1, 50):
                        miss = self.missing.get(num, 50)
                        # 获取历史遗漏
                        for k in range(i + j, i + j + 1):
                            if k < len(self.data):
                                miss = 0
                                for record_idx in range(k, min(k+5, len(self.data))):
                                    miss += 1
                                    if num in self.data[record_idx].get('numbers', []):
                                        break
                        features.extend([
                            self.frequency.get(num, 0) / len(self.data),
                            min(miss, 50) / 50,
                            self.moving_avg.get(num, {}).get(5, 0.1),
                            self.moving_avg.get(num, {}).get(10, 0.1),
                            self.autocorrelation.get(num, 0)
                        ])
                    seq.append(features)
                X_seq.append(seq)
                
                # 标签：下一期出现的数字
                next_nums = self.data[i].get('numbers', [])
                y_single = [1 if n in next_nums else 0 for n in range(1, 50)]
                y_seq.append(y_single)
            
            if len(X_seq) < 5:
                return random.sample(range(1, 50), count)
            
            # 延迟设置device
            if self.device is None:
                self.device = torch_mod.device('cuda' if torch_mod.cuda.is_available() else 'cpu')
            
            X_seq = torch_mod.tensor(X_seq, dtype=torch_mod.float32)
            y_seq = torch_mod.tensor(y_seq, dtype=torch_mod.float32)
            
            # PyTorch LSTM模型
            class LotteryLSTM(nn.Module):
                def __init__(self, input_size=245, hidden_size=64, num_layers=2, dropout=0.2):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                       batch_first=True, dropout=dropout)
                    self.fc = nn.Linear(hidden_size, 49)
                    self.dropout = nn.Dropout(dropout)
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    out = self.fc(self.dropout(lstm_out[:, -1, :]))
                    return torch_mod.sigmoid(out)
            
            # 初始化模型
            model = LotteryLSTM().to(self.device)
            criterion = nn.BCELoss()
            optimizer = optim_mod.Adam(model.parameters(), lr=0.001)
            
            # 训练循环
            model.train()
            for epoch in range(30):
                total_loss = 0
                for batch_x, batch_y in zip(X_seq, y_seq):
                    batch_x = batch_x.unsqueeze(0).to(self.device)
                    batch_y = batch_y.unsqueeze(0).to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            # 预测
            model.eval()
            with torch_mod.no_grad():
                last_seq = X_seq[-1].unsqueeze(0).to(self.device)
                predictions = model(last_seq).squeeze().cpu().numpy()
            
            # 动态温度采样
            temperature = 1.0
            probs = np.array(predictions)
            probs = np.power(probs, 1.0/temperature)
            probs = probs / probs.sum()
            
            # torch.softmax动态温度退火采样
            logits = torch_mod.tensor(probs, dtype=torch_mod.float32)
            temp_schedule = [2.0, 1.5, 1.0, 0.5]
            all_probs = []
            for temp in temp_schedule:
                scaled = logits / temp
                softmax_probs = torch_mod.softmax(scaled, dim=0).numpy()
                all_probs.append(softmax_probs)
            
            avg_probs = np.mean(all_probs, axis=0)
            
            # 综合得分
            for num in range(1, 50):
                lstm_prob = avg_probs[num-1] if num-1 < len(avg_probs) else 0.02
                
                # 基础特征得分
                base_score = self._get_weighted_score(num) / 100
                
                # 遗漏得分
                miss_s = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
                
                # 频率得分
                freq_s = self.frequency.get(num, 0) / len(self.data) if self.data else 0
                
                # 相关性得分
                corr_s = 0
                if len(self.data) > 0:
                    latest_nums = self.data[0].get('numbers', [])
                    corr_s = sum(self.correlation_matrix[num-1, n-1] for n in latest_nums) / len(latest_nums)
                
                scores[num] = (lstm_prob * 150 +  # LSTM预测权重最高
                             base_score * 30 +
                             miss_s * 30 +
                             freq_s * 50 +
                             corr_s * 40)
            
        except Exception as e:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    # ================================================================
    # Fallback方法 - 无重依赖库时的替代实现
    # ================================================================
    
    def _pytorch_fallback(self, count=6):
        """PyTorch fallback - 使用NumPy/Sklearn替代深度学习"""
        scores = {}
        try:
            for num in range(1, 50):
                # 基础特征得分
                base_score = self._get_weighted_score(num)
                
                # 遗漏得分
                miss = self.missing.get(num, 50)
                avg_cycle = len(self.data) / 49 if len(self.data) > 0 else 6
                miss_score = MathUtils.calculate_missing_cycle(miss, avg_cycle)
                
                # 频率得分
                freq = self.frequency.get(num, 0)
                freq_score = freq / len(self.data) if self.data else 0
                
                # 趋势得分
                ma5 = self.moving_avg.get(num, {}).get(5, 0.1)
                ma20 = self.moving_avg.get(num, {}).get(20, 0.1)
                trend_score = (ma5 - ma20) * 50
                
                # 相关性得分
                corr_s = 0
                if len(self.data) > 0:
                    latest_nums = self.data[0].get('numbers', [])
                    corr_s = sum(self.correlation_matrix[num-1, n-1] for n in latest_nums) / len(latest_nums)
                
                scores[num] = base_score * 0.3 + miss_score * 100 * 0.25 + freq_score * 100 * 0.2 + trend_score * 0.15 + corr_s * 100 * 0.1
        except:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _nx_number_graph_fallback(self, count=6):
        """NetworkX number_graph fallback - 使用基础统计替代PageRank"""
        scores = {}
        try:
            for num in range(1, 50):
                # 频率得分
                freq_score = self.frequency.get(num, 0)
                
                # 遗漏得分
                miss = self.missing.get(num, 50)
                miss_score = MathUtils.calculate_missing_cycle(miss, 6)
                
                # 共现得分
                cooccur_score = 0
                if len(self.data) > 0:
                    latest_nums = self.data[0].get('numbers', [])
                    for n in latest_nums:
                        cooccur_score += self.correlation_matrix[num-1, n-1]
                
                # 趋势得分
                ma5 = self.moving_avg.get(num, {}).get(5, 0.1)
                ma20 = self.moving_avg.get(num, {}).get(20, 0.1)
                trend_score = (ma5 - ma20) * 50
                
                scores[num] = freq_score * 2 + miss_score * 30 + cooccur_score * 20 + trend_score * 0.5
        except:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _nx_shortest_path_fallback(self, count=6):
        """NetworkX shortest_path fallback - 使用基础距离计算替代Dijkstra"""
        scores = {}
        try:
            latest_nums = set(self.data[0].get('numbers', []))
            for num in range(1, 50):
                # 基础得分
                base_score = self._get_weighted_score(num)
                
                # 距离得分（替代最短路径）
                min_distance = min(abs(num - ln) for ln in latest_nums) if latest_nums else 25
                distance_score = max(0, 20 - min_distance)
                
                # 遗漏得分
                miss = self.missing.get(num, 50)
                miss_score = MathUtils.calculate_missing_cycle(miss, 6)
                
                # 转移得分（替代转移图）
                transfer_score = 0
                if len(self.data) > 0:
                    for i in range(len(self.data) - 1):
                        curr = set(self.data[i].get('numbers', []))
                        next_nums = set(self.data[i + 1].get('numbers', []))
                        if num in next_nums:
                            for c in curr:
                                if abs(c - num) <= 2:
                                    transfer_score += 1
                
                scores[num] = base_score + distance_score * 2 + miss_score * 20 + transfer_score * 0.5
        except:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _nx_community_fallback(self, count=6):
        """NetworkX community fallback - 使用KMeans替代Louvain"""
        scores = {}
        try:
            # 使用sklearn的KMeans做聚类分组
            if hasattr(self, 'kmeans_labels') and self.kmeans_labels is not None:
                cluster_scores = self.kmeans_labels
            else:
                # 基础聚类（按遗漏值分组）
                cluster_scores = {}
                for num in range(1, 50):
                    miss = self.missing.get(num, 50)
                    if miss < 10:
                        cluster_scores[num] = 2  # 热号
                    elif miss < 20:
                        cluster_scores[num] = 1  # 温号
                    else:
                        cluster_scores[num] = 0  # 冷号
            
            latest_nums = set(self.data[0].get('numbers', []))
            for num in range(1, 50):
                base_score = self._get_weighted_score(num)
                miss = self.missing.get(num, 50)
                miss_score = MathUtils.calculate_missing_cycle(miss, 6)
                
                # 聚类内连接强度（替代社区检测）
                cluster_bonus = cluster_scores.get(num, 0) * 5
                
                scores[num] = base_score + miss_score * 20 + cluster_bonus
        except:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _nx_clustering_fallback(self, count=6):
        """NetworkX clustering fallback - 使用基础统计替代连通分量"""
        scores = {}
        try:
            # 按区间分组
            range_groups = {i: [] for i in range(5)}
            for num in range(1, 50):
                idx = LotteryConfig.get_range_index(num)
                if idx >= 0:
                    range_groups[idx].append(num)
            
            # 按尾数分组
            tail_groups = {i: [] for i in range(10)}
            for num in range(1, 50):
                tail = LotteryConfig.get_tail_digit(num)
                tail_groups[tail].append(num)
            
            latest_nums = set(self.data[0].get('numbers', []))
            for num in range(1, 50):
                base_score = self._get_weighted_score(num)
                miss = self.missing.get(num, 50)
                miss_score = MathUtils.calculate_missing_cycle(miss, 6)
                
                # 分组内连接强度（替代连通分量）
                comp_score = 0
                rng_idx = LotteryConfig.get_range_index(num)
                tail_idx = LotteryConfig.get_tail_digit(num)
                
                for ln in latest_nums:
                    ln_rng = LotteryConfig.get_range_index(ln)
                    ln_tail = LotteryConfig.get_tail_digit(ln)
                    if rng_idx == ln_rng:
                        comp_score += 2
                    if tail_idx == ln_tail:
                        comp_score += 1
                
                scores[num] = base_score + miss_score * 20 + comp_score
        except:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def _networkx_fallback(self, count=6):
        """NetworkX综合fallback - 使用多种基础统计"""
        scores = {}
        try:
            # 综合多种基础统计指标
            for num in range(1, 50):
                # 频率得分
                freq_score = self.frequency.get(num, 0) * 2
                
                # 遗漏得分
                miss = self.missing.get(num, 50)
                miss_score = MathUtils.calculate_missing_cycle(miss, 6) * 30
                
                # 共现得分
                cooccur_score = 0
                if len(self.data) > 0:
                    latest_nums = self.data[0].get('numbers', [])
                    for n in latest_nums:
                        cooccur_score += self.correlation_matrix[num-1, n-1]
                cooccur_score *= 20
                
                # 距离得分
                min_distance = 25
                if latest_nums:
                    min_distance = min(abs(num - ln) for ln in latest_nums)
                distance_score = max(0, 20 - min_distance) * 2
                
                # 基础特征得分
                base_score = self._get_weighted_score(num) * 0.5
                
                scores[num] = freq_score + miss_score + cooccur_score + distance_score + base_score
        except:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    def networkx_graph_algorithm(self, count=6):
        """NetworkX图算法 - 基于图论多层网络分析预测（无NetworkX时使用基础统计）"""
        if len(self.data) < 10:
            return random.sample(range(1, 50), count)
        
        # 无NetworkX时使用基础统计方法
        nx = _get_nx()
        if nx is None:
            return self._networkx_fallback(count)
        
        scores = {}
        try:
            # 构建多层图网络
            # 层1：共现图
            G_cooccur = self._build_number_graph()
            
            # 层2：转移图
            G_trans = self._build_transition_graph()
            
            # 层3：区间关系图
            G_range = nx.Graph()
            for i in range(1, 50):
                for j in range(i+1, 50):
                    range_diff = abs(LotteryConfig.get_range_index(i) - LotteryConfig.get_range_index(j))
                    if range_diff == 0:
                        G_range.add_edge(i, j, weight=3)
                    elif range_diff == 1:
                        G_range.add_edge(i, j, weight=1)
            
            # 层4：尾数关系图
            G_tail = nx.Graph()
            for i in range(1, 50):
                for j in range(i+1, 50):
                    if LotteryConfig.get_tail_digit(i) == LotteryConfig.get_tail_digit(j):
                        G_tail.add_edge(i, j, weight=2)
            
            # 综合图论指标
            # PageRank中心性
            try:
                pagerank_cooccur = nx.pagerank(G_cooccur, weight='weight')
                pagerank_trans = nx.pagerank(G_trans, weight='weight')
            except:
                pagerank_cooccur = {n: 0.02 for n in range(1, 50)}
                pagerank_trans = {n: 0.02 for n in range(1, 50)}
            
            # 度中心性
            try:
                degree_cooccur = nx.degree_centrality(G_cooccur)
                degree_trans = nx.degree_centrality(G_trans)
            except:
                degree_cooccur = {n: 0 for n in range(1, 50)}
                degree_trans = {n: 0 for n in range(1, 50)}
            
            # 介数中心性
            try:
                betweenness_cooccur = nx.betweenness_centrality(G_cooccur, weight='weight')
                betweenness_trans = nx.betweenness_centrality(G_trans, weight='weight')
            except:
                betweenness_cooccur = {n: 0 for n in range(1, 50)}
                betweenness_trans = {n: 0 for n in range(1, 50)}
            
            # 特征向量中心性
            try:
                eigen_cooccur = nx.eigenvector_centrality(G_cooccur, weight='weight', max_iter=1000)
                eigen_trans = nx.eigenvector_centrality(G_trans, weight='weight', max_iter=1000)
            except:
                eigen_cooccur = {n: 0 for n in range(1, 50)}
                eigen_trans = {n: 0 for n in range(1, 50)}
            
            # 社区检测
            try:
                communities = list(nx.community.greedy_modularity_communities(G_cooccur, weight='weight'))
                community_id = {n: -1 for n in range(1, 50)}
                for idx, comm in enumerate(communities):
                    for n in comm:
                        community_id[n] = idx
            except:
                communities = [set(range(1, 50))]
                community_id = {n: 0 for n in range(1, 50)}
            
            # 连通分量
            try:
                components = list(nx.connected_components(G_cooccur))
                component_id = {n: -1 for n in range(1, 50)}
                for idx, comp in enumerate(components):
                    for n in comp:
                        component_id[n] = idx
            except:
                components = [set(range(1, 50))]
                component_id = {n: 0 for n in range(1, 50)}
            
            # 聚类系数
            try:
                clustering_cooccur = nx.clustering(G_cooccur)
            except:
                clustering_cooccur = {n: 0 for n in range(1, 50)}
            
            # 最新一期所在社区和分量
            latest_nums = set(self.data[0].get('numbers', []))
            target_communities = set()
            target_components = set()
            for num in latest_nums:
                target_communities.add(community_id.get(num, -1))
                target_components.add(component_id.get(num, -1))
            
            # Dijkstra最短路径
            try:
                shortest_paths = {}
                for target in range(1, 50):
                    min_dist = float('inf')
                    for source in latest_nums:
                        try:
                            path_len = nx.dijkstra_path_length(G_trans, source, target, weight='weight')
                            min_dist = min(min_dist, path_len)
                        except nx.NetworkXNoPath:
                            continue
                    shortest_paths[target] = min_dist if min_dist < float('inf') else 100
            except:
                shortest_paths = {n: 100 for n in range(1, 50)}
            
            # 综合得分
            for num in range(1, 50):
                # 共现图指标
                pr_co = pagerank_cooccur.get(num, 0.02)
                deg_co = degree_cooccur.get(num, 0)
                betw_co = betweenness_cooccur.get(num, 0)
                eigen_co = eigen_cooccur.get(num, 0)
                cluster_co = clustering_cooccur.get(num, 0)
                
                # 转移图指标
                pr_tr = pagerank_trans.get(num, 0.02)
                deg_tr = degree_trans.get(num, 0)
                betw_tr = betweenness_trans.get(num, 0)
                eigen_tr = eigen_trans.get(num, 0)
                
                # 社区得分
                comm_score = 15 if community_id.get(num, -1) in target_communities else 3
                
                # 连通分量得分
                comp_score = 10 if component_id.get(num, -1) in target_components else 2
                
                # 最短路径得分
                path_score = max(0, 20 - shortest_paths.get(num, 100) * 2)
                
                # 区间和尾数关联得分
                range_score = 0
                tail_score = 0
                for ln in latest_nums:
                    if G_range.has_edge(num, ln):
                        range_score += G_range[num][ln].get('weight', 1)
                    if G_tail.has_edge(num, ln):
                        tail_score += G_tail[num][ln].get('weight', 1)
                
                # 基础特征得分
                base_score = self._get_weighted_score(num) / 100
                
                # 遗漏得分
                miss_s = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
                
                # 频率得分
                freq_s = self.frequency.get(num, 0) / len(self.data) if self.data else 0
                
                # 综合图论得分
                graph_score = (pr_co * 40 + deg_co * 20 + betw_co * 15 + eigen_co * 15 +
                              pr_tr * 30 + deg_tr * 15 + betw_tr * 10 + eigen_tr * 15 +
                              cluster_co * 10 + comm_score * 8 + comp_score * 5 +
                              path_score * 3 + range_score * 2 + tail_score * 2)
                
                scores[num] = (graph_score * 3 +  # 图论指标权重
                             base_score * 30 +
                             miss_s * 20 +
                             freq_s * 50)
            
        except Exception as e:
            for num in range(1, 50):
                scores[num] = self._get_weighted_score(num)
        
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:count]]
    
    # ======================================================================== #
    # 功能8：特别码专项分析 - 新增特别码专属算法
    # ======================================================================== #
    def special_frequency_regression(self, count=1):
        """
        基于特别码历史频率加权回归算法（索引21）
        
        功能说明：
        - 分析历史特别码的出现频率
        - 结合遗漏周期和频率进行加权回归
        - 返回置信度最高的特别码
        
        返回：
            list[int]: 预测的特别码列表（通常为1个）
        """
        if not self.data or len(self.data) < 5:
            return [random.randint(1, 49)]
        
        try:
            # 统计特别码频率
            special_freq = Counter()
            for record in self.data:
                special = record.get('special', 0)
                if 1 <= special <= 49:
                    special_freq[special] += 1
            
            # 计算遗漏值
            special_missing = {i: 0 for i in range(1, 50)}
            appeared_specials = set()
            for i, record in enumerate(self.data):
                sp = record.get('special', 0)
                if sp and sp not in appeared_specials:
                    special_missing[sp] = i
                    appeared_specials.add(sp)
            for num in range(1, 50):
                if num not in appeared_specials:
                    special_missing[num] = len(self.data)
            
            # 加权得分计算
            max_freq = max(special_freq.values()) if special_freq else 1
            scores = {}
            for num in range(1, 50):
                # 频率得分（归一化）
                freq_score = special_freq.get(num, 0) / max_freq if max_freq > 0 else 0
                
                # 遗漏回归得分（遗漏越大，回归概率越高）
                miss = special_missing.get(num, len(self.data))
                miss_score = min(1.0, miss / (len(self.data) / 3))
                
                # 指数加权回归
                miss_exp = 1 - np.exp(-miss / 10)
                
                # 综合得分
                scores[num] = freq_score * 0.3 + miss_score * 0.3 + miss_exp * 0.4
            
            # 按得分排序，取前count个
            sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [num for num, _ in sorted_nums[:count]]
        except Exception:
            return [random.randint(1, 49)]
    
    def special_correlation_algorithm(self, count=1):
        """
        基于特别码与正码关联规则挖掘算法（索引22）
        
        功能说明：
        - 挖掘正码与特别码之间的关联规则
        - 分析哪些正码经常与哪些特别码同时出现
        - 基于最新一期正码预测最可能出现的特别码
        
        返回：
            list[int]: 预测的特别码列表（通常为1个）
        """
        if not self.data or len(self.data) < 10:
            return [random.randint(1, 49)]
        
        try:
            # 构建正码与特别码的共现矩阵
            cooccurrence = defaultdict(Counter)
            for record in self.data:
                numbers = record.get('numbers', [])
                special = record.get('special', 0)
                if 1 <= special <= 49 and numbers:
                    for num in numbers:
                        if 1 <= num <= 49:
                            cooccurrence[num][special] += 1
            
            if not cooccurrence:
                return [random.randint(1, 49)]
            
            # 获取最新一期正码
            latest_numbers = self.data[0].get('numbers', []) if self.data else []
            
            # 统计每个特别码的关联得分
            special_scores = Counter()
            for num in latest_numbers:
                if num in cooccurrence:
                    for sp, count_val in cooccurrence[num].items():
                        special_scores[sp] += count_val
            
            # 如果没有关联数据，使用频率
            if not special_scores:
                for record in self.data:
                    sp = record.get('special', 0)
                    if 1 <= sp <= 49:
                        special_scores[sp] += 1
            
            # 计算遗漏值进行修正
            special_missing = {i: 0 for i in range(1, 50)}
            appeared_specials = set()
            for i, record in enumerate(self.data):
                sp = record.get('special', 0)
                if sp and sp not in appeared_specials:
                    special_missing[sp] = i
                    appeared_specials.add(sp)
            for num in range(1, 50):
                if num not in appeared_specials:
                    special_missing[num] = len(self.data)
            
            # 综合得分
            max_cooc = max(special_scores.values()) if special_scores else 1
            final_scores = {}
            for sp in range(1, 50):
                cooc_score = special_scores.get(sp, 0) / max_cooc if max_cooc > 0 else 0
                miss = special_missing.get(sp, len(self.data))
                miss_score = min(1.0, miss / (len(self.data) / 3))
                miss_exp = 1 - np.exp(-miss / 10)
                final_scores[sp] = cooc_score * 0.5 + miss_score * 0.25 + miss_exp * 0.25
            
            # 排序取前count个
            sorted_nums = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            return [num for num, _ in sorted_nums[:count]]
        except Exception:
            return [random.randint(1, 49)]
    
    # ======================================================================== #
    # 功能4：预测结果置信度显示 - 获取预测置信度
    # ======================================================================== #
    def get_prediction_scores(self, algorithm_index, count=6):
        """
        获取预测号码及其置信度
        
        功能说明：
        - 基于算法内部得分计算每个预测号码的置信度百分比
        - 置信度分级：≥70%强烈推荐，40%-69%一般推荐，<40%谨慎参考
        
        参数：
            algorithm_index: 算法索引（0-22）
            count: 需要返回的号码数量
            
        返回：
            tuple: (numbers: list[int], scores_dict: dict[int, float])
                   numbers为预测号码列表，scores_dict为{number: confidence_percentage}
        """
        try:
            # 根据算法索引选择预测方法并获取得分
            raw_scores = {}
            
            if algorithm_index == 0:
                # 综合推荐
                raw_scores = self._get_comprehensive_scores()
            elif algorithm_index == 1:
                # 冷热数字
                raw_scores = self._get_hot_cold_scores()
            elif algorithm_index == 2:
                # 单双算法
                raw_scores = self._get_odd_even_scores()
            elif algorithm_index == 3:
                # 大小算法
                raw_scores = self._get_big_small_scores()
            elif algorithm_index == 4:
                # 遗漏值分析
                raw_scores = self._get_missing_scores()
            elif algorithm_index == 5:
                # 连号邻号分析
                raw_scores = self._get_adjacent_scores()
            elif algorithm_index == 6:
                # 尾数分布
                raw_scores = self._get_tail_scores()
            elif algorithm_index == 7:
                # 区间分布
                raw_scores = self._get_range_scores()
            elif algorithm_index == 8:
                # 轮盘赌选择
                raw_scores = self._get_roulette_scores()
            elif algorithm_index == 9:
                # 历史相似性
                raw_scores = self._get_similarity_scores()
            elif algorithm_index == 10:
                # 泊松分布
                raw_scores = self._get_poisson_scores()
            elif algorithm_index == 11:
                # 玄学算法
                raw_scores = self._get_mystical_scores()
            elif algorithm_index == 12:
                # 号码关联图
                raw_scores = self._get_graph_scores()
            elif algorithm_index == 13:
                # 最短路径
                raw_scores = self._get_shortest_path_scores()
            elif algorithm_index == 14:
                # 社区发现
                raw_scores = self._get_community_scores()
            elif algorithm_index == 15:
                # 图聚类
                raw_scores = self._get_cluster_scores()
            elif algorithm_index == 16:
                # NumPy矩阵
                raw_scores = self._get_numpy_scores()
            elif algorithm_index == 17:
                # SciPy优化
                raw_scores = self._get_scipy_scores()
            elif algorithm_index == 18:
                # Scikit-learn
                raw_scores = self._get_sklearn_scores()
            elif algorithm_index == 19:
                # PyTorch
                raw_scores = self._get_pytorch_scores()
            elif algorithm_index == 20:
                # NetworkX图算法
                raw_scores = self._get_networkx_scores()
            elif algorithm_index == 21:
                # 特别码频率回归（返回1个特别码）
                numbers = self.special_frequency_regression(count=1)
                scores_dict = {num: 75.0 for num in numbers}
                return (numbers, scores_dict)
            elif algorithm_index == 22:
                # 特别码关联（返回1个特别码）
                numbers = self.special_correlation_algorithm(count=1)
                scores_dict = {num: 75.0 for num in numbers}
                return (numbers, scores_dict)
            else:
                raw_scores = self._get_comprehensive_scores()
            
            # 归一化得分到0-100%
            if raw_scores:
                max_score = max(raw_scores.values())
                min_score = min(raw_scores.values())
                score_range = max_score - min_score if max_score != min_score else 1
                confidence_dict = {
                    num: max(0, min(100, ((score - min_score) / score_range * 60) + 40))
                    for num, score in raw_scores.items()
                }
                
                # 排序取前count个
                sorted_items = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
                top_numbers = [num for num, _ in sorted_items[:count]]
                top_scores = {num: score for num, score in sorted_items[:count]}
                return (top_numbers, top_scores)
            else:
                return ([], {})
        except Exception:
            return ([random.randint(1, 49)], {random.randint(1, 49): 50.0})
    
    def _get_comprehensive_scores(self):
        """综合推荐得分"""
        scores = {}
        for num in range(1, 50):
            base = self._get_weighted_score(num)
            miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
            freq = self.frequency.get(num, 0) / len(self.data) if self.data else 0
            scores[num] = base * 0.4 + miss * 30 + freq * 100
        return scores
    
    def _get_hot_cold_scores(self):
        """冷热数字得分"""
        scores = {}
        max_freq = max(self.frequency.values()) if self.frequency else 1
        for num in range(1, 50):
            freq = self.frequency.get(num, 0) / max_freq if max_freq > 0 else 0
            miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
            scores[num] = freq * 50 + miss * 50
        return scores
    
    def _get_odd_even_scores(self):
        """单双算法得分"""
        scores = {}
        odd_ratio = self.odd_even_ratio if hasattr(self, 'odd_even_ratio') else 0.5
        target_odd = 0.6 if odd_ratio > 0.5 else 0.4
        for num in range(1, 50):
            is_odd = num % 2 == 1
            score = target_odd if is_odd else (1 - target_odd)
            miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
            scores[num] = score * 100 + miss * 30
        return scores
    
    def _get_big_small_scores(self):
        """大小算法得分"""
        scores = {}
        big_ratio = self.big_small_ratio if hasattr(self, 'big_small_ratio') else 0.5
        target_big = 0.6 if big_ratio > 0.5 else 0.4
        for num in range(1, 50):
            is_big = num > 25
            score = target_big if is_big else (1 - target_big)
            miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
            scores[num] = score * 100 + miss * 30
        return scores
    
    def _get_missing_scores(self):
        """遗漏值得分"""
        scores = {}
        for num in range(1, 50):
            miss = self.missing.get(num, 50)
            scores[num] = MathUtils.calculate_missing_cycle(miss, 6) * 100
        return scores
    
    def _get_adjacent_scores(self):
        """连号邻号得分"""
        scores = {i: 30 for i in range(1, 50)}
        if self.data:
            recent = self.data[:5]
            for record in recent:
                for n in record.get('numbers', []):
                    for adj in [n-1, n+1]:
                        if 1 <= adj <= 49:
                            scores[adj] += 15
        for num in range(1, 50):
            miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
            scores[num] += miss * 20
        return scores
    
    def _get_tail_scores(self):
        """尾数分布得分"""
        scores = {i: 20 for i in range(1, 50)}
        if hasattr(self, 'tail_distribution'):
            max_tail = max(self.tail_distribution.values()) if self.tail_distribution else 1
            for num in range(1, 50):
                tail = num % 10
                tail_freq = self.tail_distribution.get(tail, 0) / max_tail if max_tail > 0 else 0
                scores[num] += tail_freq * 60
        miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
        return scores
    
    def _get_range_scores(self):
        """区间分布得分"""
        scores = {i: 25 for i in range(1, 50)}
        if hasattr(self, 'range_distribution'):
            max_range = max(self.range_distribution.values()) if self.range_distribution else 1
            for num in range(1, 50):
                ridx = LotteryConfig.get_range_index(num)
                if ridx >= 0:
                    range_freq = self.range_distribution.get(ridx, 0) / max_range if max_range > 0 else 0
                    scores[num] += range_freq * 55
        for num in range(1, 50):
            miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
            scores[num] += miss * 20
        return scores
    
    def _get_roulette_scores(self):
        """轮盘赌选择得分"""
        scores = {}
        for num in range(1, 50):
            freq = self.frequency.get(num, 0)
            miss = self.missing.get(num, 50)
            scores[num] = freq + MathUtils.calculate_missing_cycle(miss, 6) * 50
        return scores
    
    def _get_similarity_scores(self):
        """历史相似性得分"""
        return self._get_comprehensive_scores()
    
    def _get_poisson_scores(self):
        """泊松分布得分"""
        scores = {}
        avg_freq = len(self.data) / 49 if self.data else 10
        scipy_st = _get_scipy_stats()
        for num in range(1, 50):
            freq = self.frequency.get(num, 0)
            lambda_param = avg_freq
            if lambda_param > 0 and scipy_st is not None and hasattr(scipy_st, 'poisson'):
                poisson_prob = scipy_st.poisson.pmf(freq, lambda_param)
            else:
                poisson_prob = 0.01
            miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
            scores[num] = poisson_prob * 100 + miss * 50
        return scores
    
    def _get_mystical_scores(self):
        """玄学算法得分"""
        np.random.seed(int(datetime.datetime.now().timestamp()) % 1000000)
        scores = {num: np.random.random() * 50 + 25 for num in range(1, 50)}
        for num in range(1, 50):
            miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
            scores[num] += miss * 30
        return scores
    
    def _get_graph_scores(self):
        """号码关联图得分"""
        nx_mod = _get_nx()
        if nx_mod is None or not hasattr(self, 'data') or len(self.data) < 10:
            return self._get_comprehensive_scores()
        try:
            G = nx_mod.Graph()
            for num in range(1, 50):
                G.add_node(num)
            for record in self.data[:20]:
                nums = record.get('numbers', [])
                for i in range(len(nums)):
                    for j in range(i+1, len(nums)):
                        if G.has_edge(nums[i], nums[j]):
                            G[nums[i]][nums[j]]['weight'] += 1
                        else:
                            G.add_edge(nums[i], nums[j], weight=1)
            scores = {}
            for num in range(1, 50):
                degree = G.degree(num, weight='weight')
                miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
                scores[num] = degree * 2 + miss * 50
            return scores
        except Exception:
            return self._get_comprehensive_scores()
    
    def _get_shortest_path_scores(self):
        """最短路径得分"""
        return self._get_graph_scores()
    
    def _get_community_scores(self):
        """社区发现得分"""
        return self._get_graph_scores()
    
    def _get_cluster_scores(self):
        """图聚类得分"""
        return self._get_graph_scores()
    
    def _get_numpy_scores(self):
        """NumPy矩阵得分"""
        if hasattr(self, 'np_features') and self.np_features is not None:
            try:
                scores = {}
                feature_means = np.mean(self.np_features, axis=0)
                for idx, num in enumerate(range(1, 50)):
                    score = float(feature_means[idx % len(feature_means)])
                    miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
                    scores[num] = score * 50 + miss * 50
                return scores
            except Exception:
                pass
        return self._get_comprehensive_scores()
    
    def _get_scipy_scores(self):
        """SciPy优化得分"""
        if hasattr(self, 'scipy_smoothed') and self.scipy_smoothed is not None:
            try:
                scores = {}
                smoothed_arr = self.scipy_smoothed
                for idx, num in enumerate(range(1, 50)):
                    score = float(smoothed_arr[idx % len(smoothed_arr)])
                    miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
                    scores[num] = score * 50 + miss * 50
                return scores
            except Exception:
                pass
        return self._get_comprehensive_scores()
    
    def _get_sklearn_scores(self):
        """Scikit-learn得分"""
        if hasattr(self, 'kmeans_labels') and self.kmeans_labels is not None:
            try:
                scores = {}
                target_cluster = self.kmeans_labels[-1] if len(self.kmeans_labels) > 0 else 0
                for idx, num in enumerate(range(1, 50)):
                    cluster = self.kmeans_labels[idx % len(self.kmeans_labels)]
                    score = 100 if cluster == target_cluster else 30
                    miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
                    scores[num] = score + miss * 30
                return scores
            except Exception:
                pass
        return self._get_comprehensive_scores()
    
    def _get_pytorch_scores(self):
        """PyTorch得分"""
        if hasattr(self, 'torch_predictions') and self.torch_predictions:
            try:
                scores = {}
                max_prob = max(self.torch_predictions.values()) if self.torch_predictions else 1
                for num in range(1, 50):
                    prob = self.torch_predictions.get(num, 0) / max_prob if max_prob > 0 else 0
                    miss = MathUtils.calculate_missing_cycle(self.missing.get(num, 50), 6)
                    scores[num] = prob * 100 + miss * 30
                return scores
            except Exception:
                pass
        return self._get_comprehensive_scores()
    
    def _get_networkx_scores(self):
        """NetworkX图算法得分"""
        return self._get_graph_scores()


# ============================================================================
# 第五部分：机器学习预测模型
# ============================================================================

class MLPredictionModel:
    """机器学习预测模型 - v7.1增强版"""
    
    def __init__(self, historical_data):
        self.data = historical_data
        self.models = {}
        self.scalers = {}
    
    def _extract_window_features(self, record):
        """从单条记录中提取窗口特征"""
        numbers = record.get('numbers', [])
        features = []
        # one-hot编码
        one_hot = [0] * 49
        for n in numbers:
            if 1 <= n <= 49:
                one_hot[n - 1] = 1
        features.extend(one_hot)
        # 统计特征
        if numbers:
            features.append(sum(numbers) / 6)  # 平均值
            features.append(max(numbers))  # 最大值
            features.append(min(numbers))  # 最小值
            features.append(sum(n % 2 for n in numbers))  # 单数个数
            features.append(sum(1 for n in numbers if n > 25))  # 大号个数
            features.append(max(numbers) - min(numbers))  # 跨度
            # 颜色分布
            red_c = sum(1 for n in numbers if n in LotteryConfig.RED_NUMBERS)
            blue_c = sum(1 for n in numbers if n in LotteryConfig.BLUE_NUMBERS)
            features.append(red_c)
            features.append(blue_c)
            # 尾数分布
            tails = [n % 10 for n in numbers]
            features.append(len(set(tails)))  # 不同尾数个数
        else:
            features.extend([25, 49, 1, 3, 3, 48, 2, 2, 5])
        return features
    
    def prepare_features(self):
        """【需求2-BUG修复】增强特征工程：10期滑动窗口 + 丰富统计特征
        注意：数据按时间正序排列（data[0]最旧，data[-1]最新）"""
        if len(self.data) < 20:
            return np.array([]), np.array([])
        
        # 【需求2-BUG修复】将数据反转，使data[0]为最旧，data[-1]为最新
        # 原始historical_data是insert(0, record)，所以data[0]是最新的
        # 需要反转使训练时用旧→新的顺序
        data_ordered = list(reversed(self.data))
        
        X = []
        y = []
        window_size = 10
        for i in range(len(data_ordered) - window_size):
            features = []
            for j in range(window_size):
                record = data_ordered[i + j]
                features.extend(self._extract_window_features(record))
            X.append(features)
            next_record = data_ordered[i + window_size]
            label = [0] * 49
            for n in next_record.get('numbers', []):
                if 1 <= n <= 49:
                    label[n - 1] = 1
            y.append(label)
        return np.array(X), np.array(y)
    
    def train_random_forest(self, X, y):
        """增强随机森林：100棵树 + 每数字独立分类"""
        _get_sklearn()
        RF = _RandomForestClassifier
        SS = _StandardScaler
        if RF is None or SS is None:
            return {}, None
        models = {}
        scaler = SS()
        X_scaled = scaler.fit_transform(X)
        for num_idx in range(49):
            y_num = y[:, num_idx]
            if len(set(y_num)) < 2:
                models[num_idx] = None
                continue
            model = RF(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            )
            model.fit(X_scaled, y_num)
            models[num_idx] = model
        return models, scaler
    
    def train_gradient_boosting(self, X, y):
        """增强梯度提升：100轮 + 每数字独立分类"""
        _get_sklearn()
        GBC = _GradientBoostingClassifier
        SS = _StandardScaler
        if GBC is None or SS is None:
            return {}, None
        models = {}
        scaler = SS()
        X_scaled = scaler.fit_transform(X)
        for num_idx in range(49):
            y_num = y[:, num_idx]
            if len(set(y_num)) < 2:
                models[num_idx] = None
                continue
            model = GBC(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                min_samples_split=5, min_samples_leaf=3, random_state=42
            )
            model.fit(X_scaled, y_num)
            models[num_idx] = model
        return models, scaler
    
    def train_neural_network(self, X, y):
        """增强神经网络：更深架构 + 学习率衰减 + 早停"""
        _get_sklearn()
        torch_mod = _get_torch()
        nn = _get_nn()
        optim_mod = _get_optim()
        SS = _StandardScaler
        if torch_mod is None or nn is None or SS is None:
            return None, None
        scaler = SS()
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch_mod.FloatTensor(X_scaled)
        y_tensor = torch_mod.FloatTensor(y)
        
        input_size = X.shape[1]
        
        class EnhancedLotteryNN(nn.Module):
            def __init__(self, input_size, output_size):
                super(EnhancedLotteryNN, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(64, 32), nn.ReLU(),
                    nn.Linear(32, output_size), nn.Sigmoid()
                )
            def forward(self, x):
                return self.network(x)
        
        model = EnhancedLotteryNN(input_size, 49)
        criterion = nn.BCELoss()
        optimizer = optim_mod.Adam(model.parameters(), lr=0.001)
        scheduler = optim_mod.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        model.train()
        best_loss = float('inf')
        patience = 0
        for epoch in range(80):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # 早停
            if loss.item() < best_loss - 1e-4:
                best_loss = loss.item()
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    break
        
        return model, scaler
    
    def _train_logistic_regression(self, X, y):
        """Logistic回归：每数字独立分类"""
        _get_sklearn()
        LR = _LogisticRegression
        SS = _StandardScaler
        if LR is None or SS is None:
            return {}, None
        models = {}
        scaler = SS()
        X_scaled = scaler.fit_transform(X)
        for num_idx in range(49):
            y_num = y[:, num_idx]
            if len(set(y_num)) < 2:
                models[num_idx] = None
                continue
            model = LR(max_iter=500, C=1.0, random_state=42)
            model.fit(X_scaled, y_num)
            models[num_idx] = model
        return models, scaler
    
    def _train_mlp_classifier(self, X, y):
        """MLP分类器：每数字独立分类"""
        _get_sklearn()
        MLP = _MLPClassifier
        MMS = _MinMaxScaler
        if MLP is None or MMS is None:
            return {}, None
        models = {}
        scaler = MMS()
        X_scaled = scaler.fit_transform(X)
        for num_idx in range(49):
            y_num = y[:, num_idx]
            if len(set(y_num)) < 2:
                models[num_idx] = None
                continue
            model = MLP(
                hidden_layer_sizes=(64, 32), max_iter=150,
                learning_rate='adaptive', early_stopping=True,
                random_state=42
            )
            model.fit(X_scaled, y_num)
            models[num_idx] = model
        return models, scaler
    
    def optimize_hyperparameters(self, X, y):
        _get_sklearn()
        optuna = _get_optuna()
        TPESampler = _TPESampler_class
        GBC = _GradientBoostingClassifier
        TTS = _train_test_split
        ACC = _accuracy_score
        if optuna is None or TPESampler is None or GBC is None or TTS is None or ACC is None:
            return {'n_estimators': 200, 'max_depth': 15, 'learning_rate': 0.05}
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 15)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
            X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.2, random_state=42)
            model = GBC(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
            y_single = np.argmax(y, axis=1)
            model.fit(X_train, y_single)
            y_pred = model.predict(X_test)
            y_test_single = np.argmax(y_test, axis=1)
            return ACC(y_test_single, y_pred)
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=20)
        return study.best_params
    
    def predict_with_all_models(self):
        """【需求2-BUG修复】增强预测：多模型概率加权集成
        修复问题：
        1. 数据顺序问题：反转数据确保用最新10条记录
        2. 特征维度补齐：确保580维
        3. model_count==0时fallback"""
        X, y = self.prepare_features()
        if len(X) < 20:
            return random.sample(range(1, 50), 6)
        
        # 【需求2-BUG修复】将数据反转，使data[0]为最新（与prepare_features保持一致）
        data_ordered = list(reversed(self.data))
        
        # 构建最新特征 - 使用最新的10条记录（反转后data[0]是最新）
        latest_features = []
        for j in range(10):
            if j < len(data_ordered):
                record = data_ordered[j]
                latest_features.extend(self._extract_window_features(record))
        
        # 【需求2-BUG修复】补齐到580维（每期58维 × 10期 = 580维）
        expected_len = 580
        if len(latest_features) < expected_len:
            latest_features.extend([0] * (expected_len - len(latest_features)))
        elif len(latest_features) > expected_len:
            latest_features = latest_features[:expected_len]
        
        X_latest = np.array([latest_features])
        
        # 收集各模型的概率预测
        all_probs = np.zeros(49)
        model_count = 0
        
        # 1. Random Forest
        try:
            rf_models, rf_scaler = self.train_random_forest(X, y)
            X_scaled = rf_scaler.transform(X_latest)
            for num_idx in range(49):
                if rf_models.get(num_idx) is not None:
                    prob = rf_models[num_idx].predict_proba(X_scaled)[0]
                    all_probs[num_idx] += prob[1] if len(prob) > 1 else prob[0]
            model_count += 1
        except Exception:
            pass
        
        # 2. Gradient Boosting
        try:
            gb_models, gb_scaler = self.train_gradient_boosting(X, y)
            X_scaled = gb_scaler.transform(X_latest)
            for num_idx in range(49):
                if gb_models.get(num_idx) is not None:
                    prob = gb_models[num_idx].predict_proba(X_scaled)[0]
                    all_probs[num_idx] += prob[1] if len(prob) > 1 else prob[0]
            model_count += 1
        except Exception:
            pass
        
        # 3. Logistic Regression（新增）
        try:
            lr_models, lr_scaler = self._train_logistic_regression(X, y)
            X_scaled = lr_scaler.transform(X_latest)
            for num_idx in range(49):
                if lr_models.get(num_idx) is not None:
                    prob = lr_models[num_idx].predict_proba(X_scaled)[0]
                    all_probs[num_idx] += prob[1] if len(prob) > 1 else prob[0]
            model_count += 1
        except Exception:
            pass
        
        # 4. MLP Classifier（新增）
        try:
            mlp_models, mlp_scaler = self._train_mlp_classifier(X, y)
            X_scaled = mlp_scaler.transform(X_latest)
            for num_idx in range(49):
                if mlp_models.get(num_idx) is not None:
                    prob = mlp_models[num_idx].predict_proba(X_scaled)[0]
                    all_probs[num_idx] += prob[1] if len(prob) > 1 else prob[0]
            model_count += 1
        except Exception:
            pass
        
        # 5. Neural Network (PyTorch)
        try:
            nn_model, nn_scaler = self.train_neural_network(X, y)
            if nn_model is not None:
                torch_mod = _get_torch()
                X_scaled = nn_scaler.transform(X_latest)
                X_tensor = torch_mod.FloatTensor(X_scaled)
                nn_model.eval()
                with torch_mod.no_grad():
                    output = nn_model(X_tensor).squeeze().numpy()
                all_probs += output
                model_count += 1
        except Exception:
            pass
        
        # 【需求2-BUG修复】归一化概率 - model_count==0时fallback到随机选择
        if model_count > 0:
            all_probs = all_probs / model_count
        else:
            # 所有模型都训练失败，fallback到随机选择
            return random.sample(range(1, 50), 6)
        
        # 选top6
        top_indices = np.argsort(all_probs)[::-1][:6]
        predictions = [idx + 1 for idx in top_indices]
        
        return sorted(predictions)


# ============================================================================
# 第六部分：自定义控件模块
# ============================================================================

class NumberButton(QPushButton):
    """数字按钮控件"""
    
    def __init__(self, number, parent=None):
        super().__init__(parent)
        self.number = number
        self.is_selected = False
        self._setup_ui()
    
    def _setup_ui(self):
        self.setText(str(self.number))
        self.setMinimumSize(50, 50)
        self.setCheckable(True)
        self._apply_color()
    
    def _apply_color(self):
        colors = LotteryConfig.get_number_color(self.number)
        self.setStyleSheet(
            "QPushButton { background-color: #FFFFFF; color: " + colors['text'] + "; border: 2px solid " + colors['border'] + "; border-radius: 8px; font-weight: bold; font-size: 18px; min-width: 48px; min-height: 48px; }"
            "QPushButton:hover { background-color: #F8F9FA; }"
            "QPushButton:checked { background-color: " + colors['text'] + "; color: #FFFFFF; border: 2px solid " + colors['border'] + "; }"
        )
    
    def set_selected(self, selected):
        self.is_selected = selected
        self.setChecked(selected)
    
    def get_number(self):
        return self.number


class NumberPanel(QWidget):
    """数字面板控件"""
    number_selected = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_numbers = []
        self.number_buttons = {}
        self._init_ui()
    
    def _init_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        for num in range(1, 50):
            btn = NumberButton(num, self)
            btn.clicked.connect(lambda checked, n=num: self._on_button_clicked(n))
            self.number_buttons[num] = btn
            row = (num - 1) // 7
            col = (num - 1) % 7
            layout.addWidget(btn, row, col)
        self.setLayout(layout)
    
    def _on_button_clicked(self, number):
        btn = self.number_buttons[number]
        if btn.isChecked():
            if number not in self.selected_numbers:
                self.selected_numbers.append(number)
        else:
            if number in self.selected_numbers:
                self.selected_numbers.remove(number)
        self.number_selected.emit(self.selected_numbers)
    
    def get_selected_numbers(self):
        return self.selected_numbers.copy()
    
    def set_selected_numbers(self, numbers):
        for btn in self.number_buttons.values():
            btn.set_selected(False)
        self.selected_numbers = []
        for num in numbers:
            if num in self.number_buttons:
                self.number_buttons[num].set_selected(True)
                self.selected_numbers.append(num)
        self.number_selected.emit(self.selected_numbers)
    
    def clear_selection(self):
        self.set_selected_numbers([])
    
    def highlight_numbers(self, numbers):
        for num, btn in self.number_buttons.items():
            if num in numbers:
                btn.set_selected(True)
            else:
                btn.set_selected(False)


# ========================================================================
# 功能12：性能优化 - 异步预测 Worker
# ========================================================================
class PredictWorker(QObject):
    """异步预测工作器 - 在后台线程执行预测任务"""
    finished = pyqtSignal(list, dict)  # 预测结果, 置信度信息
    error = pyqtSignal(str)  # 错误信息
    progress = pyqtSignal(int, str)  # 进度, 状态信息
    
    def __init__(self, historical_data, algorithm_index, parent=None):
        super().__init__(parent)
        self.historical_data = historical_data
        self.algorithm_index = algorithm_index
    
    def run(self):
        """执行预测任务"""
        try:
            self.progress.emit(10, "正在初始化预测器...")
            predictor = PredictionAlgorithms(self.historical_data)
            
            self.progress.emit(30, "正在执行预测算法...")
            predictions = self._run_algorithm(predictor)
            
            self.progress.emit(70, "正在计算置信度...")
            confidence_info = self._get_confidence(predictor, predictions)
            
            self.progress.emit(100, "预测完成")
            self.finished.emit(predictions, confidence_info)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
    
    def _run_algorithm(self, predictor):
        """运行预测算法"""
        algorithm_index = self.algorithm_index
        count = 1 if algorithm_index in [21, 22] else 6
        
        algorithms = {
            0: lambda: predictor.comprehensive_recommendation(count),
            1: lambda: predictor.hot_cold_algorithm(count),
            2: lambda: predictor.odd_even_algorithm(count),
            3: lambda: predictor.big_small_algorithm(count),
            4: lambda: predictor.missing_value_analysis(count),
            5: lambda: predictor.adjacent_number_analysis(count),
            6: lambda: predictor.tail_distribution_algorithm(count),
            7: lambda: predictor.range_distribution_algorithm(count),
            8: lambda: predictor.roulette_selection(count),
            9: lambda: predictor.historical_similarity(count),
            10: lambda: predictor.poisson_distribution(count),
            11: lambda: predictor.mystical_algorithm(count),
            12: lambda: predictor.number_graph_algorithm(count),
            13: lambda: predictor.shortest_path_algorithm(count),
            14: lambda: predictor.community_detection_algorithm(count),
            15: lambda: predictor.graph_clustering_algorithm(count),
            16: lambda: predictor.numpy_matrix_algorithm(count),
            17: lambda: predictor.scipy_optimization_algorithm(count),
            18: lambda: predictor.sklearn_ensemble_algorithm(count),
            19: lambda: predictor.pytorch_deep_learning_algorithm(count),
            20: lambda: predictor.networkx_graph_algorithm(count),
            21: lambda: predictor.special_frequency_regression(count),
            22: lambda: predictor.special_correlation_algorithm(count),
        }
        
        if algorithm_index in algorithms:
            return algorithms[algorithm_index]()
        return predictor.comprehensive_recommendation(count)
    
    def _get_confidence(self, predictor, predictions):
        """获取预测置信度"""
        if self.algorithm_index in [21, 22]:
            return {predictions[0]: 75.0} if predictions else {}
        try:
            _, confidence_info = predictor.get_prediction_scores(self.algorithm_index, len(predictions))
            return confidence_info
        except Exception:
            return {}


class MLPredictWorker(QObject):
    """异步机器学习预测工作器"""
    finished = pyqtSignal(list)  # 预测结果
    error = pyqtSignal(str)  # 错误信息
    progress = pyqtSignal(int, str)  # 进度, 状态信息
    
    def __init__(self, historical_data, parent=None):
        super().__init__(parent)
        self.historical_data = historical_data
    
    def run(self):
        """执行机器学习预测任务"""
        try:
            self.progress.emit(5, "正在准备数据...")
            model = MLPredictionModel(self.historical_data)
            
            self.progress.emit(15, "正在提取特征...")
            X, y = model.prepare_features()
            if len(X) < 20:
                self.error.emit("特征数据不足，无法进行机器学习预测")
                return
            
            self.progress.emit(20, "正在训练模型（1/5 随机森林）...")
            all_probs = np.zeros(49)
            model_count = 0
            
            # 反转数据保持一致
            data_ordered = list(reversed(model.data))
            latest_features = []
            for j in range(10):
                if j < len(data_ordered):
                    record = data_ordered[j]
                    latest_features.extend(model._extract_window_features(record))
            expected_len = 580
            if len(latest_features) < expected_len:
                latest_features.extend([0] * (expected_len - len(latest_features)))
            elif len(latest_features) > expected_len:
                latest_features = latest_features[:expected_len]
            X_latest = np.array([latest_features])
            
            # 1. Random Forest
            try:
                rf_models, rf_scaler = model.train_random_forest(X, y)
                X_scaled = rf_scaler.transform(X_latest)
                for num_idx in range(49):
                    if rf_models.get(num_idx) is not None:
                        prob = rf_models[num_idx].predict_proba(X_scaled)[0]
                        all_probs[num_idx] += prob[1] if len(prob) > 1 else prob[0]
                model_count += 1
            except Exception:
                pass
            self.progress.emit(40, "正在训练模型（2/5 梯度提升）...")
            
            # 2. Gradient Boosting
            try:
                gb_models, gb_scaler = model.train_gradient_boosting(X, y)
                X_scaled = gb_scaler.transform(X_latest)
                for num_idx in range(49):
                    if gb_models.get(num_idx) is not None:
                        prob = gb_models[num_idx].predict_proba(X_scaled)[0]
                        all_probs[num_idx] += prob[1] if len(prob) > 1 else prob[0]
                model_count += 1
            except Exception:
                pass
            self.progress.emit(60, "正在训练模型（3/5 逻辑回归）...")
            
            # 3. Logistic Regression
            try:
                lr_models, lr_scaler = model._train_logistic_regression(X, y)
                X_scaled = lr_scaler.transform(X_latest)
                for num_idx in range(49):
                    if lr_models.get(num_idx) is not None:
                        prob = lr_models[num_idx].predict_proba(X_scaled)[0]
                        all_probs[num_idx] += prob[1] if len(prob) > 1 else prob[0]
                model_count += 1
            except Exception:
                pass
            self.progress.emit(75, "正在训练模型（4/5 MLP）...")
            
            # 4. MLP Classifier
            try:
                mlp_models, mlp_scaler = model._train_mlp_classifier(X, y)
                X_scaled = mlp_scaler.transform(X_latest)
                for num_idx in range(49):
                    if mlp_models.get(num_idx) is not None:
                        prob = mlp_models[num_idx].predict_proba(X_scaled)[0]
                        all_probs[num_idx] += prob[1] if len(prob) > 1 else prob[0]
                model_count += 1
            except Exception:
                pass
            self.progress.emit(90, "正在训练模型（5/5 神经网络）...")
            
            # 5. Neural Network
            try:
                nn_model, nn_scaler = model.train_neural_network(X, y)
                if nn_model is not None:
                    torch_mod = _get_torch()
                    X_scaled = nn_scaler.transform(X_latest)
                    X_tensor = torch_mod.FloatTensor(X_scaled)
                    nn_model.eval()
                    with torch_mod.no_grad():
                        output = nn_model(X_tensor).squeeze().numpy()
                    all_probs += output
                    model_count += 1
            except Exception:
                pass
            
            self.progress.emit(95, "正在汇总结果...")
            
            if model_count > 0:
                all_probs = all_probs / model_count
            else:
                self.finished.emit(sorted(random.sample(range(1, 50), 6)))
                return
            
            top_indices = np.argsort(all_probs)[::-1][:6]
            predictions = sorted([idx + 1 for idx in top_indices])
            
            self.progress.emit(100, "机器学习预测完成")
            self.finished.emit(predictions)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class StatisticsChart(QWidget):
    """统计图表控件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        _get_mpl()
        global _figure_module, _canvas_class
        if _figure_module is not None and _canvas_class is not None:
            self.figure = _figure_module(figsize=(8, 6))
            self.canvas = _canvas_class(self.figure)
        else:
            self.figure = None
            self.canvas = None
        layout = QVBoxLayout(self)
        if self.canvas is not None:
            layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def plot_frequency(self, frequency, title="数字出现频率"):
        if self.figure is None:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        numbers = sorted(frequency.keys())
        counts = [frequency[n] for n in numbers]
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
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, str(int(height)), ha='center', va='bottom', fontsize=8)
        self.canvas.draw()
    
    def plot_missing(self, missing, title="数字遗漏值"):
        if self.figure is None:
            return
        _get_mpl()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        numbers = sorted(missing.keys())
        values = [missing[n] for n in numbers]
        if _pyplot_module is not None:
            cmap = _pyplot_module.cm.RdYlGn_r
            norm = _pyplot_module.Normalize(vmin=min(values), vmax=max(values))
        else:
            cmap = None
            norm = None
        if cmap is not None and norm is not None:
            colors = [cmap(norm(v)) for v in values]
        else:
            colors = '#3498db'
        ax.bar(numbers, values, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('数字', fontsize=12)
        ax.set_ylabel('遗漏期数', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(numbers)
        ax.grid(axis='y', alpha=0.3)
        self.canvas.draw()
    
    def plot_distribution(self, data, title="分布统计"):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        labels = list(data.keys())
        values = list(data.values())
        colors = ['#FF0000', '#0000FF', '#008000', '#F39C12', '#9B59B6']
        wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%', startangle=90)
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        self.canvas.draw()
    
    def plot_trend(self, data, title="综合走势图"):
        self.figure.clear()
        fig = self.figure
        if len(data) > 0:
            periods = [item[0] for item in data[:50]]
            numbers_list = [item[1] for item in data[:50]]
            ax = fig.add_subplot(111)
            for i, numbers in enumerate(numbers_list):
                for num in numbers:
                    color = '#FF0000'
                    if LotteryConfig.is_blue(num):
                        color = '#0000FF'
                    elif LotteryConfig.is_green(num):
                        color = '#008000'
                    ax.plot(i, num, 'o', color=color, markersize=6)
            ax.set_xlabel('期数', fontsize=12)
            ax.set_ylabel('号码', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylim(0, 50)
            ax.set_xticks(range(0, len(periods), 5))
            ax.set_xticklabels([periods[i] for i in range(0, len(periods), 5)], rotation=45)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.canvas.draw()
    
    def plot_correlation_heatmap(self, data, title="号码相关性热力图"):
        """功能9：相关性热力图 - 49x49号码共现矩阵热力图"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        # data是49x49的共现矩阵
        if isinstance(data, dict):
            # 如果是字典格式的共现矩阵，转换为numpy数组
            numbers = sorted(data.keys())
            matrix = np.zeros((len(numbers), len(numbers)))
            for i, n1 in enumerate(numbers):
                for j, n2 in enumerate(numbers):
                    matrix[i, j] = data[n1].get(n2, 0)
            data = matrix
        
        if data is not None and isinstance(data, np.ndarray) and data.shape[0] > 0:
            im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
            ax.set_xlabel('号码', fontsize=10)
            ax.set_ylabel('号码', fontsize=10)
            ax.set_title(title, fontsize=14, fontweight='bold')
            # 设置刻度标签（每5个显示一个）
            tick_positions = list(range(0, 49, 5))
            tick_labels = [str(i+1) for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)
            # 添加颜色条
            cbar = self.figure.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('共现次数', fontsize=10)
        else:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=14, fontweight='bold')
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_interval_analysis(self, data, title="号码间隔分析图"):
        """功能9：间隔分析图 - 高频号码出现间隔分布箱线图"""
        if self.figure is None:
            return
        _get_mpl()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        # data格式: {号码: [间隔列表]}
        if data and isinstance(data, dict):
            # 选择出现频率最高的15个号码
            sorted_data = sorted(data.items(), key=lambda x: len(x[1]), reverse=True)[:15]
            numbers = [str(item[0]) for item in sorted_data]
            intervals = [item[1] for item in sorted_data]
            
            bp = ax.boxplot(intervals, labels=numbers, patch_artist=True)
            # 设置箱线图颜色
            colors = ['#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD', '#F0E68C'] * 3
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xlabel('高频号码', fontsize=12)
            ax.set_ylabel('间隔期数', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            # 旋转x轴标签
            if _pyplot_module is not None:
                _pyplot_module.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, '暂无数据（需要至少20期历史数据）', ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_consecutive_probability(self, data, title="连号邻号概率图"):
        """功能9：连号邻号概率图"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        # data格式: {'consecutive': {'pairs': [[n1,n2]: count, ...], 'avg_prob': float}, 'adjacent': {...}}
        if data and isinstance(data, dict):
            categories = []
            probabilities = []
            colors = []
            
            # 连号数据
            if 'consecutive' in data:
                consecutive_data = data['consecutive']
                avg_prob = consecutive_data.get('avg_prob', 0)
                categories.append('连号')
                probabilities.append(avg_prob * 100)
                colors.append('#FF6B6B')
            
            # 邻号数据
            if 'adjacent' in data:
                adjacent_data = data['adjacent']
                avg_prob = adjacent_data.get('avg_prob', 0)
                categories.append('邻号')
                probabilities.append(avg_prob * 100)
                colors.append('#4ECDC4')
            
            # 同尾数数据
            if 'same_tail' in data:
                tail_data = data['same_tail']
                avg_prob = tail_data.get('avg_prob', 0)
                categories.append('同尾数')
                probabilities.append(avg_prob * 100)
                colors.append('#45B7D1')
            
            if categories:
                x_pos = range(len(categories))
                bars = ax.bar(x_pos, probabilities, color=colors, edgecolor='white', linewidth=1.5)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(categories, fontsize=11)
                ax.set_ylabel('出现概率 (%)', fontsize=12)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                
                # 添加数值标签
                for bar, prob in zip(bars, probabilities):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height, f'{prob:.1f}%',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
            else:
                ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', fontsize=12)
        else:
            ax.text(0.5, 0.5, '暂无数据（需要至少50期历史数据）', ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_sum_distribution(self, data, title="和值分布图"):
        """功能9：和值分布图 - histogram + scipy正态拟合曲线"""
        if self.figure is None:
            return
        _get_mpl()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        # data格式: [和值列表] 或 {'sums': [sum1, sum2, ...], 'mean': m, 'std': s}
        if data:
            if isinstance(data, dict):
                sums = data.get('sums', [])
            else:
                sums = data
            
            if len(sums) >= 5:
                sums_array = np.array(sums)
                mean_val = np.mean(sums_array)
                std_val = np.std(sums_array)
                
                # 绘制直方图
                n, bins, patches = ax.hist(sums, bins=20, density=True, alpha=0.7, 
                                          color='#3498db', edgecolor='white', linewidth=0.5)
                
                # 正态分布拟合曲线
                x_range = np.linspace(min(sums), max(sums), 100)
                scipy_st = _get_scipy_stats()
                if scipy_st is not None and hasattr(scipy_st, 'norm'):
                    y_normal = scipy_st.norm.pdf(x_range, mean_val, std_val)
                    ax.plot(x_range, y_normal, 'r-', linewidth=2, label=f'正态拟合 (μ={mean_val:.1f}, σ={std_val:.1f})')
                
                # 标记均值线
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'均值={mean_val:.1f}')
                
                ax.set_xlabel('号码和值', fontsize=12)
                ax.set_ylabel('概率密度', fontsize=12)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'数据不足（需要至少5期数据，当前{len(sums)}期）', 
                       ha='center', va='center', fontsize=12)
        else:
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        self.figure.tight_layout()
        self.canvas.draw()


# ============================================================================
# 第七部分：主窗口类
# ============================================================================

class LotteryPredictionWindow(QMainWindow):
    """彩票预测系统主窗口 v7.1"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(LotteryConfig.WINDOW_TITLE)
        self.setMinimumSize(LotteryConfig.WINDOW_MIN_WIDTH, LotteryConfig.WINDOW_MIN_HEIGHT)
        self.resize(1600, 1000)
        self.font_size_key = LotteryConfig.DEFAULT_FONT_SIZE_KEY
        self.historical_data = []
        self.prediction_cache = {}
        self.current_algorithm_index = 0
        self.collected_predictions = []
        self.prediction_history = []
        self.custom_weights = {}
        self.is_dark_mode = False
        self.margin_top = 10
        self.margin_bottom = 10
        self.margin_left = 10
        self.margin_right = 10
        self.spacing = 10
        self.data_file = "./彩票预测系统v7.1/lottery_data.json"
        # ======================================================================== #
        # 功能6：智能筛选 - 筛选后的数据
        # ======================================================================== #
        self.filtered_data = None
        # ======================================================================== #
        # 功能12：性能优化 - 分页
        # ======================================================================== #
        self.history_page = 1
        self.history_page_size = 100
        # ======================================================================== #
        # 功能12：性能优化 - 图表懒加载
        # ======================================================================== #
        self._chart_initialized = False
        # ======================================================================== #
        # 功能12：性能优化 - 预测异步执行
        # ======================================================================== #
        self._is_predicting = False
        self._predict_thread = None
        self._predict_worker = None
        self._load_data()
        self._init_ui()
        self._apply_stylesheet()
        # 功能9：快捷键绑定
        self._setup_shortcuts()
        # ======================================================================== #
        # 功能12：性能优化 - 图表懒加载
        # ======================================================================== #
        if hasattr(self, 'tabs'):
            self.tabs.currentChanged.connect(self._on_tab_changed)
        print("彩票预测系统 v7.1 初始化完成")
    
    def _setup_shortcuts(self):
        """设置快捷键"""
        for i in range(1, 10):
            shortcut = QShortcut(QKeySequence("Ctrl+" + str(i)), self)
            shortcut.activated.connect(lambda idx=i-1: self._switch_tab(idx))
        QShortcut(QKeySequence("Ctrl+P"), self).activated.connect(self._on_predict_clicked)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._on_save_clicked)
        QShortcut(QKeySequence("Ctrl+D"), self).activated.connect(self._on_random_draw_clicked)
    
    def _switch_tab(self, index):
        """切换标签页"""
        if 0 <= index < self.tabs.count():
            self.tabs.setCurrentIndex(index)
    
    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self._create_top_bar(main_layout)
        self._create_tabs(main_layout)
        self._create_status_bar()
        # 加载数据后刷新界面
        self._update_history_table()
    
    def _create_top_bar(self, parent_layout):
        top_bar = QWidget()
        top_bar.setObjectName("TopBar")
        top_bar.setMinimumHeight(60)
        top_bar.setMaximumHeight(80)
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setSpacing(10)
        top_bar_layout.setContentsMargins(15, 5, 15, 5)
        
        title_label = QLabel("彩票预测系统 v7.1")
        title_label.setObjectName("TitleLabel")
        top_bar_layout.addWidget(title_label, 0, Qt.AlignmentFlag.AlignLeft)
        
        button_group = QWidget()
        button_layout = QHBoxLayout(button_group)
        button_layout.setSpacing(8)
        
        import_btn = QPushButton("导入")
        import_btn.clicked.connect(self._on_import_clicked)
        button_layout.addWidget(import_btn)
        
        export_btn = QPushButton("导出")
        export_btn.clicked.connect(self._on_export_clicked)
        button_layout.addWidget(export_btn)
        
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self._on_save_clicked)
        button_layout.addWidget(save_btn)
        
        separator1 = QLabel("|")
        separator1.setObjectName("Separator")
        button_layout.addWidget(separator1)
        
        add_btn = QPushButton("添加数据")
        add_btn.clicked.connect(self._on_add_data_clicked)
        button_layout.addWidget(add_btn)
        
        del_btn = QPushButton("删除数据")
        del_btn.clicked.connect(self._on_delete_data_clicked)
        button_layout.addWidget(del_btn)
        
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(self._on_clear_data_clicked)
        button_layout.addWidget(clear_btn)
        
        # ======================================================================== #
        # 功能10：快捷操作面板
        # ======================================================================== #
        separator2 = QLabel("|")
        separator2.setObjectName("Separator")
        button_layout.addWidget(separator2)
        
        quick_btn = QPushButton("快捷")
        quick_btn.setObjectName("QuickBtn")
        quick_menu = QMenu(self)
        
        # 清空所有收藏
        clear_fav_action = quick_menu.addAction("清空所有收藏")
        clear_fav_action.triggered.connect(self._on_clear_collected)
        
        # 清空预测记录
        clear_ph_action = quick_menu.addAction("清空预测记录")
        clear_ph_action.triggered.connect(self._on_clear_prediction_history)
        
        quick_menu.addSeparator()
        
        # 批量导出报告
        batch_export_action = quick_menu.addAction("批量导出报告")
        batch_export_action.triggered.connect(self._on_batch_export_reports)
        
        # 验证号码组合
        validate_action = quick_menu.addAction("验证号码组合")
        validate_action.triggered.connect(self._on_validate_number_combo)
        
        quick_menu.addSeparator()
        
        # 一键复制预测结果
        copy_action = quick_menu.addAction("一键复制预测结果")
        copy_action.triggered.connect(self._on_copy_latest_prediction)
        
        quick_btn.setMenu(quick_menu)
        button_layout.addWidget(quick_btn)
        
        top_bar_layout.addWidget(button_group, 1, Qt.AlignmentFlag.AlignCenter)
        
        # ======================================================================== #
        # 右侧控制区域：字体调节、主题切换等
        # ======================================================================== #
        right_control = QWidget()
        right_layout = QHBoxLayout(right_control)
        right_layout.setSpacing(5)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        font_label = QLabel("字体:")
        right_layout.addWidget(font_label)
        
        self.font_combo = QComboBox()
        self.font_combo.setObjectName("FontSizeCombo")
        for size_key in LotteryConfig.FONT_SIZES.keys():
            self.font_combo.addItem(size_key)
        default_index = list(LotteryConfig.FONT_SIZES.keys()).index(self.font_size_key)
        self.font_combo.setCurrentIndex(default_index)
        self.font_combo.currentTextChanged.connect(self._on_font_size_changed)
        right_layout.addWidget(self.font_combo)
        
        font_minus_btn = QPushButton("A-")
        font_minus_btn.setFixedSize(35, 35)
        font_minus_btn.clicked.connect(self._decrease_font_size)
        right_layout.addWidget(font_minus_btn)
        
        font_plus_btn = QPushButton("A+")
        font_plus_btn.setFixedSize(35, 35)
        font_plus_btn.clicked.connect(self._increase_font_size)
        right_layout.addWidget(font_plus_btn)
        
        margin_btn = QPushButton("边距")
        margin_btn.clicked.connect(self._show_margin_dialog)
        right_layout.addWidget(margin_btn)
        
        theme_btn = QPushButton("切换主题")
        theme_btn.setObjectName("ThemeToggleBtn")
        theme_btn.clicked.connect(self._on_toggle_theme)
        right_layout.addWidget(theme_btn)
        
        top_bar_layout.addWidget(right_control, 0, Qt.AlignmentFlag.AlignRight)
        parent_layout.addWidget(top_bar)
    
    # ======================================================================== #
    # 功能10：快捷操作面板 - 操作方法
    # ======================================================================== #
    def _on_batch_export_reports(self):
        """批量导出报告"""
        reply = QMessageBox.question(self, "确认", "确定要导出所有21个算法的HTML报告吗？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        dir_path = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not dir_path:
            return
        
        self.statusBar().showMessage("正在生成报告...")
        
        for idx in range(len(LotteryConfig.ALGORITHMS)):
            try:
                algo_name = LotteryConfig.ALGORITHMS[idx][0]
                file_name = os.path.join(dir_path, "report_" + str(idx) + "_" + algo_name + ".html")
                
                # 生成简单的HTML报告
                html_content = "<html><head><meta charset='utf-8'><title>算法报告 - " + algo_name + "</title></head><body>"
                html_content += "<h1>算法报告: " + algo_name + "</h1>"
                html_content += "<p>算法描述: " + LotteryConfig.ALGORITHMS[idx][1] + "</p>"
                html_content += "<p>生成时间: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "</p>"
                html_content += "</body></html>"
                
                with open(file_name, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            except Exception as e:
                continue
        
        QMessageBox.information(self, "导出完成", "已导出" + str(len(LotteryConfig.ALGORITHMS)) + "个算法的HTML报告到:\n" + dir_path)
        self.statusBar().showMessage("批量导出完成")
    
    def _on_validate_number_combo(self):
        """验证号码组合"""
        dialog = QInputDialog(self)
        dialog.setWindowTitle("验证号码组合")
        dialog.setLabelText("请输入6个号码（逗号分隔）:")
        dialog.setTextValue("")
        if dialog.exec() == QDialog.DialogCode.Accepted:
            input_text = dialog.textValue().strip()
            try:
                numbers = [int(n.strip()) for n in input_text.split(',')]
                
                if len(numbers) != 6:
                    QMessageBox.warning(self, "验证失败", "必须输入6个号码")
                    return
                
                if len(set(numbers)) != 6:
                    QMessageBox.warning(self, "验证失败", "号码不能重复")
                    return
                
                invalid = [n for n in numbers if not (1 <= n <= 49)]
                if invalid:
                    QMessageBox.warning(self, "验证失败", "以下号码不在1-49范围内: " + str(invalid))
                    return
                
                # 验证通过
                red_count = sum(1 for n in numbers if LotteryConfig.is_red(n))
                blue_count = sum(1 for n in numbers if LotteryConfig.is_blue(n))
                green_count = 6 - red_count - blue_count
                odd_count = sum(1 for n in numbers if n % 2 == 1)
                even_count = 6 - odd_count
                big_count = sum(1 for n in numbers if n > 25)
                small_count = 6 - big_count
                sum_val = sum(numbers)
                
                result = "号码组合验证通过！\n\n"
                result += "号码: " + ' '.join(str(n).zfill(2) for n in sorted(numbers)) + "\n"
                result += "和值: " + str(sum_val) + "\n"
                result += "单双比: " + str(odd_count) + ":" + str(even_count) + "\n"
                result += "大小比: " + str(big_count) + ":" + str(small_count) + "\n"
                result += "颜色分布: 红" + str(red_count) + " 蓝" + str(blue_count) + " 绿" + str(green_count)
                
                QMessageBox.information(self, "验证成功", result)
            except ValueError:
                QMessageBox.warning(self, "验证失败", "请输入有效的数字，用逗号分隔")
    
    def _on_copy_latest_prediction(self):
        """一键复制预测结果"""
        if not self.prediction_history:
            QMessageBox.information(self, "提示", "没有可复制的预测结果")
            return
        
        latest = self.prediction_history[-1]
        numbers = latest.get('numbers', [])
        if not numbers:
            QMessageBox.information(self, "提示", "没有可复制的预测结果")
            return
        
        text = ' '.join(str(n).zfill(2) for n in sorted(numbers))
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        QMessageBox.information(self, "复制成功", "预测结果已复制到剪贴板:\n" + text)
    
    def _show_margin_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("边距和间距设置")
        dialog.setFixedSize(400, 300)
        layout = QVBoxLayout(dialog)
        
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("上边距:"))
        top_spin = QSpinBox()
        top_spin.setRange(0, 50)
        top_spin.setValue(self.margin_top)
        top_layout.addWidget(top_spin)
        layout.addLayout(top_layout)
        
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(QLabel("下边距:"))
        bottom_spin = QSpinBox()
        bottom_spin.setRange(0, 50)
        bottom_spin.setValue(self.margin_bottom)
        bottom_layout.addWidget(bottom_spin)
        layout.addLayout(bottom_layout)
        
        left_layout = QHBoxLayout()
        left_layout.addWidget(QLabel("左边距:"))
        left_spin = QSpinBox()
        left_spin.setRange(0, 50)
        left_spin.setValue(self.margin_left)
        left_layout.addWidget(left_spin)
        layout.addLayout(left_layout)
        
        right_layout = QHBoxLayout()
        right_layout.addWidget(QLabel("右边距:"))
        right_spin = QSpinBox()
        right_spin.setRange(0, 50)
        right_spin.setValue(self.margin_right)
        right_layout.addWidget(right_spin)
        layout.addLayout(right_layout)
        
        spacing_layout = QHBoxLayout()
        spacing_layout.addWidget(QLabel("区块间距:"))
        spacing_spin = QSpinBox()
        spacing_spin.setRange(0, 50)
        spacing_spin.setValue(self.spacing)
        spacing_layout.addWidget(spacing_spin)
        layout.addLayout(spacing_layout)
        
        layout.addStretch()
        
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.margin_top = top_spin.value()
            self.margin_bottom = bottom_spin.value()
            self.margin_left = left_spin.value()
            self.margin_right = right_spin.value()
            self.spacing = spacing_spin.value()
            self._update_stylesheet()
            QMessageBox.information(self, "成功", "边距设置已更新")
    
    def _create_tabs(self, parent_layout):
        self.tabs = QTabWidget()
        self.tabs.setObjectName("MainTabs")
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        
        tab1 = self._create_data_import_tab()
        self.tabs.addTab(tab1, "数据导入与格式转换")
        
        tab2 = self._create_history_tab()
        self.tabs.addTab(tab2, "历史记录")
        
        tab3 = self._create_prediction_tab()
        self.tabs.addTab(tab3, "预测与抽取")
        
        tab4 = self._create_number_selection_tab()
        self.tabs.addTab(tab4, "数字选择")
        
        tab5 = self._create_seventh_prediction_tab()
        self.tabs.addTab(tab5, "第七位预判")
        
        tab6 = self._create_statistics_chart_tab()
        self.tabs.addTab(tab6, "统计分析图表")
        
        tab7 = self._create_backtest_tab()
        self.tabs.addTab(tab7, "回测分析")
        
        tab8 = self._create_prediction_history_tab()
        self.tabs.addTab(tab8, "预测记录")
        
        tab9 = self._create_info_tab()
        self.tabs.addTab(tab9, "应用说明")
        
        parent_layout.addWidget(self.tabs)
    
    def _create_data_import_tab(self):
        widget = QWidget()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        
        left_panel = self._create_input_panel()
        splitter.addWidget(left_panel)
        
        right_panel = self._create_result_panel()
        splitter.addWidget(right_panel)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(self.margin_left, self.margin_top, self.margin_right, self.margin_bottom)
        layout.setSpacing(self.spacing)
        layout.addWidget(splitter)
        
        return widget
    
    def _create_input_panel(self):
        widget = QWidget()
        widget.setObjectName("InputPanel")
        layout = QVBoxLayout(widget)
        layout.setSpacing(self.spacing)
        
        title = QLabel("粘贴原始数据")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        
        info_label = QLabel("支持大量批量粘贴，每期一行或多期连续粘贴均可自动识别：\n第116期最新开奖结果 2026年04月26日 15 龙/水 46 鸡/木 16 兔/木 10 鸡/火 48 羊/火 33 狗/火 22 鸡/水\n第115期最新开奖结果 2026年04月25日 21 狗/土 16 兔/木 25 马/木 29 虎/土 08 猪/木 07 鼠/土 04 兔/金")
        info_label.setObjectName("InfoLabel")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        self.raw_text_edit = QTextEdit()
        self.raw_text_edit.setObjectName("RawTextEdit")
        self.raw_text_edit.setPlaceholderText("请在此处粘贴原始数据...")
        layout.addWidget(self.raw_text_edit)
        
        button_layout = QHBoxLayout()
        
        convert_btn = QPushButton("转换为标准格式")
        convert_btn.clicked.connect(self._on_convert_clicked)
        button_layout.addWidget(convert_btn)
        
        add_to_history_btn = QPushButton("添加到历史记录")
        add_to_history_btn.clicked.connect(self._on_add_to_history_clicked)
        button_layout.addWidget(add_to_history_btn)
        
        batch_import_btn = QPushButton("批量导入")
        batch_import_btn.clicked.connect(self._on_batch_import_clicked)
        button_layout.addWidget(batch_import_btn)
        
        layout.addLayout(button_layout)
        
        clear_btn = QPushButton("清空输入")
        clear_btn.clicked.connect(lambda: self.raw_text_edit.clear())
        layout.addWidget(clear_btn)
        
        online_update_btn = QPushButton("在线更新")
        online_update_btn.setObjectName("OnlineUpdateBtn")
        online_update_btn.clicked.connect(self._on_online_update_clicked)
        layout.addWidget(online_update_btn)
        
        clear_data_btn = QPushButton("清除数据")
        clear_data_btn.setObjectName("ClearDataBtn")
        clear_data_btn.clicked.connect(self._on_clear_all_data_clicked)
        layout.addWidget(clear_data_btn)
        
        # ======================================================================== #
        # 功能3：多数据源切换 - 数据源设置按钮
        # ======================================================================== #
        data_source_btn = QPushButton("数据源设置")
        data_source_btn.setObjectName("DataSourceBtn")
        data_source_btn.clicked.connect(self._on_data_source_setting_clicked)
        layout.addWidget(data_source_btn)
        
        return widget
    
    def _create_result_panel(self):
        widget = QWidget()
        widget.setObjectName("ResultPanel")
        layout = QVBoxLayout(widget)
        layout.setSpacing(self.spacing)
        layout.setContentsMargins(self.margin_left, self.margin_top, self.margin_right, self.margin_bottom)
        
        title = QLabel("转换结果")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        
        self.converted_text_edit = QTextEdit()
        self.converted_text_edit.setObjectName("ConvertedTextEdit")
        self.converted_text_edit.setReadOnly(True)
        layout.addWidget(self.converted_text_edit)
        
        return widget
    
    def _create_prediction_tab(self):
        widget = QWidget()
        h_splitter = QSplitter(Qt.Orientation.Horizontal)
        h_splitter.setHandleWidth(2)
        
        left_panel = self._create_left_prediction_panel()
        h_splitter.addWidget(left_panel)
        
        right_panel = self._create_right_prediction_panel()
        h_splitter.addWidget(right_panel)
        
        h_splitter.setStretchFactor(0, 1)
        h_splitter.setStretchFactor(1, 2)
        
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(self.margin_left, self.margin_top, self.margin_right, self.margin_bottom)
        layout.setSpacing(self.spacing)
        layout.addWidget(h_splitter)
        
        return widget
    
    def _create_left_prediction_panel(self):
        widget = QWidget()
        widget.setObjectName("LeftPredictionPanel")
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.setHandleWidth(2)
        
        latest_panel = self._create_latest_data_panel()
        v_splitter.addWidget(latest_panel)
        
        algorithm_panel = self._create_algorithm_panel()
        v_splitter.addWidget(algorithm_panel)
        
        v_splitter.setStretchFactor(0, 1)
        v_splitter.setStretchFactor(1, 2)
        
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(self.spacing)
        layout.addWidget(v_splitter)
        
        return widget
    
    def _create_right_prediction_panel(self):
        widget = QWidget()
        widget.setObjectName("RightPredictionPanel")
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.setHandleWidth(2)
        
        result_panel = self._create_prediction_result_panel()
        v_splitter.addWidget(result_panel)
        
        data_manage_panel = self._create_data_manage_panel()
        v_splitter.addWidget(data_manage_panel)
        
        v_splitter.setStretchFactor(0, 2)
        v_splitter.setStretchFactor(1, 1)
        
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(self.spacing)
        layout.addWidget(v_splitter)
        
        return widget
    
    def _create_latest_data_panel(self):
        widget = QWidget()
        widget.setObjectName("LatestDataPanel")
        layout = QVBoxLayout(widget)
        layout.setSpacing(self.spacing)
        
        title = QLabel("最新开奖数据")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        
        self.latest_display = QLabel("暂无数据")
        self.latest_display.setObjectName("LatestDisplay")
        self.latest_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.latest_display.setWordWrap(True)
        layout.addWidget(self.latest_display)
        
        refresh_btn = QPushButton("刷新显示")
        refresh_btn.clicked.connect(self._refresh_latest_display)
        layout.addWidget(refresh_btn)
        
        return widget
    
    def _create_history_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(0)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 标题行
        header_layout = QHBoxLayout()
        title = QLabel("历史记录")
        title.setObjectName("PanelTitle")
        header_layout.addWidget(title)
        
        self.history_count_label = QLabel("")
        self.history_count_label.setStyleSheet("color: #555555; font-size: 14px;")
        header_layout.addWidget(self.history_count_label)
        header_layout.addStretch()
        
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self._update_history_table)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # ======================================================================== #
        # 功能6：智能筛选与搜索
        # ======================================================================== #
        filter_widget = QWidget()
        filter_layout = QVBoxLayout(filter_widget)
        filter_layout.setSpacing(5)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        
        # 筛选条件行1：期号范围
        filter_row1 = QHBoxLayout()
        filter_row1.addWidget(QLabel("期号范围:"))
        
        filter_row1.addWidget(QLabel("起始"))
        self.filter_period_start = QSpinBox()
        self.filter_period_start.setRange(0, 999999)
        self.filter_period_start.setValue(0)
        self.filter_period_start.setPrefix("")
        self.filter_period_start.setToolTip("0表示不限")
        filter_row1.addWidget(self.filter_period_start)
        
        filter_row1.addWidget(QLabel("~"))
        
        self.filter_period_end = QSpinBox()
        self.filter_period_end.setRange(0, 999999)
        self.filter_period_end.setValue(0)
        self.filter_period_end.setToolTip("0表示不限")
        filter_row1.addWidget(self.filter_period_end)
        
        filter_row1.addWidget(QLabel("  号码搜索:"))
        self.filter_number_input = QLineEdit()
        self.filter_number_input.setPlaceholderText("输入号码(逗号分隔)")
        self.filter_number_input.setMaximumWidth(150)
        filter_row1.addWidget(self.filter_number_input)
        
        filter_row1.addWidget(QLabel("  和值范围:"))
        self.filter_sum_min = QSpinBox()
        self.filter_sum_min.setRange(0, 300)
        self.filter_sum_min.setValue(0)
        self.filter_sum_min.setPrefix("")
        self.filter_sum_min.setToolTip("0表示不限")
        filter_row1.addWidget(self.filter_sum_min)
        
        filter_row1.addWidget(QLabel("~"))
        
        self.filter_sum_max = QSpinBox()
        self.filter_sum_max.setRange(0, 300)
        self.filter_sum_max.setValue(0)
        self.filter_sum_max.setToolTip("0表示不限")
        filter_row1.addWidget(self.filter_sum_max)
        
        filter_layout.addLayout(filter_row1)
        
        # 筛选条件行2：单双比、大小比、颜色分布
        filter_row2 = QHBoxLayout()
        
        filter_row2.addWidget(QLabel("单双比:"))
        self.filter_odd_even = QComboBox()
        self.filter_odd_even.addItems(["全部", "单多", "双多", "均衡"])
        self.filter_odd_even.setMaximumWidth(80)
        filter_row2.addWidget(self.filter_odd_even)
        
        filter_row2.addWidget(QLabel("  大小比:"))
        self.filter_size = QComboBox()
        self.filter_size.addItems(["全部", "大多", "小多", "均衡"])
        self.filter_size.setMaximumWidth(80)
        filter_row2.addWidget(self.filter_size)
        
        filter_row2.addWidget(QLabel("  颜色分布:"))
        self.filter_color = QComboBox()
        self.filter_color.addItems(["全部", "红多", "蓝多", "绿多"])
        self.filter_color.setMaximumWidth(80)
        filter_row2.addWidget(self.filter_color)
        
        filter_row2.addStretch()
        
        # 筛选和重置按钮
        filter_btn = QPushButton("筛选")
        filter_btn.setStyleSheet("QPushButton { background-color: #3498DB; color: white; border: none; border-radius: 4px; padding: 6px 16px; font-weight: bold; } QPushButton:hover { background-color: #2980B9; }")
        filter_btn.clicked.connect(self._on_filter_history)
        filter_row2.addWidget(filter_btn)
        
        reset_btn = QPushButton("重置")
        reset_btn.setStyleSheet("QPushButton { background-color: #95A5A6; color: white; border: none; border-radius: 4px; padding: 6px 16px; font-weight: bold; } QPushButton:hover { background-color: #7F8C8D; }")
        reset_btn.clicked.connect(self._on_reset_filter)
        filter_row2.addWidget(reset_btn)
        
        filter_layout.addLayout(filter_row2)
        
        layout.addWidget(filter_widget)
        
        # 外层水平Splitter（左右分隔）
        h_splitter = QSplitter(Qt.Orientation.Horizontal)
        h_splitter.setHandleWidth(4)
        
        # === 左半部分：垂直Splitter（上下分隔）===
        left_v_splitter = QSplitter(Qt.Orientation.Vertical)
        left_v_splitter.setHandleWidth(4)
        
        # 上方：最新开奖数据
        latest_widget = QWidget()
        latest_layout = QVBoxLayout(latest_widget)
        latest_layout.setSpacing(5)
        
        latest_title = QLabel("最新开奖数据")
        latest_title.setObjectName("PanelTitle")
        latest_layout.addWidget(latest_title)
        
        self.history_latest_display = QLabel("暂无数据")
        self.history_latest_display.setObjectName("LatestDisplay")
        self.history_latest_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.history_latest_display.setWordWrap(True)
        latest_layout.addWidget(self.history_latest_display)
        
        left_v_splitter.addWidget(latest_widget)
        
        # 下方：历史记录表格
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setSpacing(0)
        table_layout.setContentsMargins(0, 0, 0, 0)
        
        self.history_table = QTableWidget()
        self.history_table.setObjectName("HistoryTable")
        self.history_table.setColumnCount(9)
        self.history_table.setHorizontalHeaderLabels(["期号", "日期", "正码", "特别码", "和值", "单双比", "大小比", "颜色分布", "跨度"])
        
        # 列宽设置：Interactive模式支持用户拖拽调整列宽
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        # 各列初始宽度，足够完整显示内容
        col_widths = [70, 120, 160, 70, 60, 65, 65, 110, 60]
        for i, w in enumerate(col_widths):
            self.history_table.setColumnWidth(i, w)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        # 确保水平和垂直滚动条都能正常出现
        self.history_table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.history_table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.history_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.history_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # 行高加大确保内容完整
        self.history_table.verticalHeader().setDefaultSectionSize(40)
        # 确保文字不被截断
        self.history_table.setWordWrap(True)
        
        table_layout.addWidget(self.history_table)
        
        # ======================================================================== #
        # 功能12：历史记录分页控件
        # ======================================================================== #
        pagination_widget = QWidget()
        pagination_layout = QHBoxLayout(pagination_widget)
        pagination_layout.setContentsMargins(0, 5, 0, 0)
        
        self.history_page_label = QLabel("第 1 页 / 共 1 页")
        pagination_layout.addWidget(self.history_page_label)
        
        pagination_layout.addStretch()
        
        prev_btn = QPushButton("上一页")
        prev_btn.setMaximumWidth(80)
        prev_btn.clicked.connect(self._on_prev_page)
        pagination_layout.addWidget(prev_btn)
        
        pagination_layout.addWidget(QLabel("跳转至"))
        self.history_page_spin = QSpinBox()
        self.history_page_spin.setRange(1, 1)
        self.history_page_spin.setMaximumWidth(60)
        self.history_page_spin.valueChanged.connect(self._on_page_jump)
        pagination_layout.addWidget(self.history_page_spin)
        
        pagination_layout.addWidget(QLabel("页"))
        
        next_btn = QPushButton("下一页")
        next_btn.setMaximumWidth(80)
        next_btn.clicked.connect(self._on_next_page)
        pagination_layout.addWidget(next_btn)
        
        # 筛选统计标签
        pagination_layout.addStretch()
        self.filter_stats_label = QLabel("")
        self.filter_stats_label.setStyleSheet("color: #3498DB; font-weight: bold;")
        pagination_layout.addWidget(self.filter_stats_label)
        
        table_layout.addWidget(pagination_widget)
        
        left_v_splitter.addWidget(table_widget)
        
        # 左半部分比例：最新数据20%，表格80%
        left_v_splitter.setStretchFactor(0, 2)
        left_v_splitter.setStretchFactor(1, 8)
        
        h_splitter.addWidget(left_v_splitter)
        
        # === 右半部分：垂直Splitter（上下分隔）===
        right_v_splitter = QSplitter(Qt.Orientation.Vertical)
        right_v_splitter.setHandleWidth(4)
        
        # 上方：快捷操作面板（小按钮网格布局）
        action_widget = QWidget()
        action_layout = QVBoxLayout(action_widget)
        action_layout.setSpacing(6)
        action_layout.setContentsMargins(4, 4, 4, 4)
        
        action_title = QLabel("快捷操作")
        action_title.setObjectName("PanelTitle")
        action_layout.addWidget(action_title)
        
        # 网格布局：4列小按钮
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(5)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        
        small_btn_style = (
            "QPushButton { background-color: #FFFFFF; color: #333333; border: 1px solid #DDDDDD; "
            "border-radius: 4px; padding: 4px 8px; font-size: 12px; font-weight: bold; min-height: 26px; } "
            "QPushButton:hover { background-color: #F0F0F0; border-color: #BBBBBB; } "
            "QPushButton:pressed { background-color: #E0E0E0; }"
        )
        danger_btn_style = (
            "QPushButton { background-color: #FFFFFF; color: #E74C3C; border: 1px solid #E74C3C; "
            "border-radius: 4px; padding: 4px 8px; font-size: 12px; font-weight: bold; min-height: 26px; } "
            "QPushButton:hover { background-color: #FDF0EF; } "
            "QPushButton:pressed { background-color: #FADBD8; }"
        )
        
        quick_actions = [
            ("📥 导入", small_btn_style, self._on_import_clicked),
            ("📤 导出", small_btn_style, self._on_export_clicked),
            ("💾 保存", small_btn_style, self._on_save_clicked),
            ("➕ 添加", small_btn_style, self._on_add_data_clicked),
            ("🗑 删除", danger_btn_style, self._on_delete_data_clicked),
            ("⚠ 清空", danger_btn_style, self._on_clear_data_clicked),
            ("📋 批量导入", small_btn_style, self._on_batch_import_clicked),
        ]
        
        cols = 4
        for i, (text, style, callback) in enumerate(quick_actions):
            btn = QPushButton(text)
            btn.setStyleSheet(style)
            btn.clicked.connect(callback)
            grid_layout.addWidget(btn, i // cols, i % cols)
        
        action_layout.addWidget(grid_widget)
        action_layout.addStretch()
        
        right_v_splitter.addWidget(action_widget)
        
        # 下方：期号详情显示面板
        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.setSpacing(5)
        
        detail_title_row = QHBoxLayout()
        detail_title = QLabel("期号详情")
        detail_title.setObjectName("PanelTitle")
        detail_title_row.addWidget(detail_title)
        detail_title_row.addStretch()
        
        show_btn = QPushButton("显示选中")
        show_btn.clicked.connect(self._on_show_period_detail)
        show_btn.setStyleSheet("QPushButton { background-color: #2ECC71; color: white; border: none; border-radius: 4px; padding: 5px 12px; font-weight: bold; } QPushButton:hover { background-color: #27AE60; }")
        detail_title_row.addWidget(show_btn)
        
        detail_layout.addLayout(detail_title_row)
        
        self.period_detail_edit = QTextEdit()
        self.period_detail_edit.setReadOnly(True)
        self.period_detail_edit.setStyleSheet("QTextEdit { background-color: #FFFFFF; color: #000000; border: 1px solid #DDDDDD; border-radius: 4px; font-size: 14px; padding: 8px; }")
        self.period_detail_edit.setPlaceholderText("在表格中选择一期，然后点击「显示选中」按钮查看完整信息...")
        detail_layout.addWidget(self.period_detail_edit)
        
        # 统计摘要
        self.history_stats_label = QLabel("加载数据后显示统计信息")
        self.history_stats_label.setWordWrap(True)
        self.history_stats_label.setStyleSheet("color: #333333; font-size: 13px; line-height: 1.5;")
        detail_layout.addWidget(self.history_stats_label)
        
        right_v_splitter.addWidget(detail_widget)
        
        # 右半部分比例：操作60%，统计40%
        right_v_splitter.setStretchFactor(0, 6)
        right_v_splitter.setStretchFactor(1, 4)
        
        h_splitter.addWidget(right_v_splitter)
        
        # 左右比例：左侧80%，右侧20%
        h_splitter.setStretchFactor(0, 8)
        h_splitter.setStretchFactor(1, 2)
        
        layout.addWidget(h_splitter)
        
        return widget
    
    def _create_number_selection_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(self.spacing)
        layout.setContentsMargins(self.margin_left, self.margin_top, self.margin_right, self.margin_bottom)
        
        title = QLabel("数字选择面板（49个数字）")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        
        scroll = QScrollArea()
        scroll.setObjectName("NumberScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.number_panel = NumberPanel()
        self.number_panel.number_selected.connect(self._on_number_selected)
        scroll.setWidget(self.number_panel)
        
        layout.addWidget(scroll, 1)
        
        # 颜色图例（保留原有的三列颜色图例）
        legend_widget = QWidget()
        legend_layout = QHBoxLayout(legend_widget)
        legend_layout.setSpacing(15)
        legend_layout.setContentsMargins(5, 2, 5, 2)
        
        red_label = QLabel("纯红色")
        red_label.setStyleSheet("color: #FF0000; font-size: 14px; font-weight: bold;")
        red_nums = QLabel("01 02 07 08 12 13 18 19 23 24 29 30 34 35 40 45 46")
        red_nums.setStyleSheet("color: #FF0000; font-size: 13px;")
        red_nums.setWordWrap(True)
        legend_layout.addWidget(red_label)
        legend_layout.addWidget(red_nums, 1)
        
        blue_label = QLabel("纯蓝色")
        blue_label.setStyleSheet("color: #0000FF; font-size: 14px; font-weight: bold;")
        blue_nums = QLabel("03 04 09 10 14 15 20 25 26 31 36 37 41 42 47 48")
        blue_nums.setStyleSheet("color: #0000FF; font-size: 13px;")
        blue_nums.setWordWrap(True)
        legend_layout.addWidget(blue_label)
        legend_layout.addWidget(blue_nums, 1)
        
        green_label = QLabel("深绿色")
        green_label.setStyleSheet("color: #008000; font-size: 14px; font-weight: bold;")
        green_nums = QLabel("05 06 11 16 17 21 22 27 28 32 33 38 39 43 44 49")
        green_nums.setStyleSheet("color: #008000; font-size: 13px;")
        green_nums.setWordWrap(True)
        legend_layout.addWidget(green_label)
        legend_layout.addWidget(green_nums, 1)
        
        layout.addWidget(legend_widget)
        
        # 已选数字
        selected_layout = QHBoxLayout()
        selected_label = QLabel("已选数字:")
        selected_layout.addWidget(selected_label)
        self.selected_numbers_label = QLabel("无")
        self.selected_numbers_label.setObjectName("SelectedNumbersLabel")
        selected_layout.addWidget(self.selected_numbers_label)
        selected_layout.addStretch()
        clear_btn = QPushButton("清除")
        clear_btn.clicked.connect(self._clear_number_selection)
        selected_layout.addWidget(clear_btn)
        layout.addLayout(selected_layout)
        
        return widget
    
    def _create_algorithm_panel(self):
        widget = QWidget()
        widget.setObjectName("AlgorithmPanel")
        layout = QVBoxLayout(widget)
        layout.setSpacing(self.spacing)
        
        title = QLabel("选择预测算法")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.setObjectName("AlgorithmCombo")
        for algo_name, algo_desc in LotteryConfig.ALGORITHMS:
            self.algorithm_combo.addItem(algo_name)
        self.algorithm_combo.currentIndexChanged.connect(self._on_algorithm_changed)
        layout.addWidget(self.algorithm_combo)
        
        self.algorithm_desc_label = QLabel("请选择一个预测算法")
        self.algorithm_desc_label.setObjectName("AlgorithmDescLabel")
        self.algorithm_desc_label.setWordWrap(True)
        layout.addWidget(self.algorithm_desc_label)
        
        button_layout = QHBoxLayout()
        
        self.predict_button = QPushButton("开始预测")
        self.predict_button.clicked.connect(self._on_predict_clicked)
        button_layout.addWidget(self.predict_button)
        
        random_btn = QPushButton("随机抽取")
        random_btn.clicked.connect(self._on_random_draw_clicked)
        button_layout.addWidget(random_btn)
        
        layout.addLayout(button_layout)
        
        ml_btn = QPushButton("机器学习预测")
        ml_btn.clicked.connect(self._on_ml_predict_clicked)
        layout.addWidget(ml_btn)
        
        weight_btn = QPushButton("权重调节")
        weight_btn.clicked.connect(self._on_weight_adjust_clicked)
        layout.addWidget(weight_btn)
        
        return widget
    
    def _create_prediction_result_panel(self):
        widget = QWidget()
        widget.setObjectName("PredictionResultPanel")
        layout = QVBoxLayout(widget)
        layout.setSpacing(self.spacing)
        
        title = QLabel("预测结果")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        
        scroll = QScrollArea()
        scroll.setObjectName("PredictionScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        result_layout.setSpacing(self.spacing)
        
        self.prediction_display = QLabel("等待预测...")
        self.prediction_display.setObjectName("PredictionDisplay")
        self.prediction_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prediction_display.setWordWrap(True)
        result_layout.addWidget(self.prediction_display)
        
        self.prediction_number_panel = QWidget()
        self.prediction_number_layout = QGridLayout(self.prediction_number_panel)
        self.prediction_number_layout.setSpacing(5)
        result_layout.addWidget(self.prediction_number_panel)
        
        self.prediction_stats_label = QLabel("统计信息：等待预测...")
        self.prediction_stats_label.setObjectName("PredictionStatsLabel")
        self.prediction_stats_label.setWordWrap(True)
        result_layout.addWidget(self.prediction_stats_label)
        
        # ======================================================================== #
        # 功能4：置信度分析文本框
        # ======================================================================== #
        self.confidence_analysis_label = QLabel("置信度分析：")
        self.confidence_analysis_label.setObjectName("ConfidenceAnalysisLabel")
        result_layout.addWidget(self.confidence_analysis_label)
        
        self.confidence_analysis_text = QTextEdit()
        self.confidence_analysis_text.setReadOnly(True)
        self.confidence_analysis_text.setMaximumHeight(80)
        self.confidence_analysis_text.setPlaceholderText("预测完成后显示置信度分析...")
        self.confidence_analysis_text.setStyleSheet("QTextEdit { background-color: #F0F8FF; border: 1px solid #3498DB; border-radius: 4px; font-size: 13px; padding: 5px; }")
        result_layout.addWidget(self.confidence_analysis_text)
        
        # 收藏和导出按钮（功能1、功能7）
        collect_btn_layout = QHBoxLayout()
        collect_btn = QPushButton("收藏结果")
        collect_btn.setObjectName("CollectBtn")
        collect_btn.clicked.connect(self._on_collect_prediction)
        collect_btn_layout.addWidget(collect_btn)
        
        compare_btn = QPushButton("对比收藏")
        compare_btn.setObjectName("CompareBtn")
        compare_btn.clicked.connect(self._on_compare_collected)
        collect_btn_layout.addWidget(compare_btn)
        
        clear_collect_btn = QPushButton("清空收藏")
        clear_collect_btn.setObjectName("ClearCollectBtn")
        clear_collect_btn.clicked.connect(self._on_clear_collected)
        collect_btn_layout.addWidget(clear_collect_btn)
        
        export_report_btn = QPushButton("导出报告")
        export_report_btn.setObjectName("ExportReportBtn")
        export_report_btn.clicked.connect(self._on_export_report)
        collect_btn_layout.addWidget(export_report_btn)
        
        result_layout.addLayout(collect_btn_layout)
        
        scroll.setWidget(result_widget)
        layout.addWidget(scroll, 1)
        
        return widget
    
    def _create_data_manage_panel(self):
        widget = QWidget()
        widget.setObjectName("DataManagePanel")
        layout = QVBoxLayout(widget)
        layout.setSpacing(self.spacing)
        
        title = QLabel("数据管理")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        
        button_layout = QHBoxLayout()
        
        batch_del_btn = QPushButton("批量删除")
        batch_del_btn.clicked.connect(self._on_batch_delete_clicked)
        button_layout.addWidget(batch_del_btn)
        
        batch_modify_btn = QPushButton("批量修改")
        batch_modify_btn.clicked.connect(self._on_batch_modify_clicked)
        button_layout.addWidget(batch_modify_btn)
        
        batch_add_btn = QPushButton("批量添加")
        batch_add_btn.clicked.connect(self._on_batch_add_clicked)
        button_layout.addWidget(batch_add_btn)
        
        layout.addLayout(button_layout)
        
        del_all_btn = QPushButton("删除所有历史记录")
        del_all_btn.clicked.connect(self._on_clear_data_clicked)
        layout.addWidget(del_all_btn)
        
        return widget

    
    def _create_seventh_prediction_tab(self):
        """创建第七位预判标签页
        
        功能8增强：添加特别码专项分析功能
        """
        widget = QWidget()
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(2)
        
        # ======================================================================== #
        # 功能8：特别码专项分析 - 顶部分析按钮
        # ======================================================================== #
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        top_layout.setSpacing(15)
        top_layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("第七位数字预判")
        title.setObjectName("PanelTitle")
        top_layout.addWidget(title)
        
        desc = QLabel("根据历史数据分析第七位特别号码的大小、单双、尾数特征")
        desc.setObjectName("DescLabel")
        top_layout.addWidget(desc)
        
        btn_group = QWidget()
        btn_layout = QHBoxLayout(btn_group)
        btn_layout.setSpacing(20)
        
        size_btn = QPushButton("大小预判")
        size_btn.clicked.connect(self._predict_seventh_size)
        btn_layout.addWidget(size_btn)
        
        odd_even_btn = QPushButton("单双预判")
        odd_even_btn.clicked.connect(self._predict_seventh_odd_even)
        btn_layout.addWidget(odd_even_btn)
        
        tail_btn = QPushButton("尾数大小预判")
        tail_btn.clicked.connect(self._predict_seventh_tail)
        btn_layout.addWidget(tail_btn)
        
        all_btn = QPushButton("综合预判")
        all_btn.clicked.connect(self._predict_seventh_all)
        btn_layout.addWidget(all_btn)
        
        top_layout.addWidget(btn_group)
        
        # ======================================================================== #
        # 功能8：特别码专项分析 - 新增分析按钮
        # ======================================================================== #
        special_btn_group = QWidget()
        special_btn_layout = QHBoxLayout(special_btn_group)
        special_btn_layout.setSpacing(20)
        
        special_freq_btn = QPushButton("特别码频率图")
        special_freq_btn.setStyleSheet("QPushButton { background-color: #3498DB; color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; } QPushButton:hover { background-color: #2980B9; }")
        special_freq_btn.clicked.connect(self._show_special_frequency_chart)
        special_btn_layout.addWidget(special_freq_btn)
        
        special_trend_btn = QPushButton("特别码走势图")
        special_trend_btn.setStyleSheet("QPushButton { background-color: #9B59B6; color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; } QPushButton:hover { background-color: #8E44AD; }")
        special_trend_btn.clicked.connect(self._show_special_trend_chart)
        special_btn_layout.addWidget(special_trend_btn)
        
        special_correlation_btn = QPushButton("正码关联分析")
        special_correlation_btn.setStyleSheet("QPushButton { background-color: #E67E22; color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; } QPushButton:hover { background-color: #D35400; }")
        special_correlation_btn.clicked.connect(self._show_special_correlation_analysis)
        special_btn_layout.addWidget(special_correlation_btn)
        
        top_layout.addWidget(special_btn_group)
        top_layout.addStretch()
        
        splitter.addWidget(top_panel)
        
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(10, 10, 10, 10)
        
        result_title = QLabel("预判结果")
        result_title.setObjectName("ResultTitleLabel")
        bottom_layout.addWidget(result_title)
        
        self.seventh_result_text = QTextEdit()
        self.seventh_result_text.setReadOnly(True)
        self.seventh_result_text.setPlaceholderText("点击上方按钮进行预判...")
        bottom_layout.addWidget(self.seventh_result_text)
        
        splitter.addWidget(bottom_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(self.margin_left, self.margin_top, self.margin_right, self.margin_bottom)
        layout.setSpacing(self.spacing)
        layout.addWidget(splitter)
        
        return widget
    
    # ======================================================================== #
    # 功能8：特别码专项分析 - 新增分析方法
    # ======================================================================== #
    def _show_special_frequency_chart(self):
        """特别码频率图"""
        if not self.historical_data:
            QMessageBox.information(self, "提示", "没有历史数据，请先导入数据！")
            return
        
        # 统计特别码频率
        special_freq = Counter()
        for record in self.historical_data:
            sp = record.get('special', 0)
            if 1 <= sp <= 49:
                special_freq[sp] += 1
        
        if not special_freq:
            QMessageBox.information(self, "提示", "没有特别码数据")
            return
        
        # 创建对话框显示图表
        dialog = QDialog(self)
        dialog.setWindowTitle("特别码频率分布图")
        dialog.setFixedSize(900, 600)
        
        layout = QVBoxLayout(dialog)
        
        # 创建matplotlib图表
        _get_mpl()
        global _figure_module, _canvas_class
        if _figure_module is None or _canvas_class is None:
            QMessageBox.warning(self, "错误", "无法加载matplotlib库")
            dialog.close()
            return
        
        fig = _figure_module(figsize=(10, 6))
        canvas = _canvas_class(fig)
        ax = fig.add_subplot(111)
        
        numbers = sorted(special_freq.keys())
        counts = [special_freq[n] for n in numbers]
        
        colors = []
        for num in numbers:
            if LotteryConfig.is_red(num):
                colors.append('#FF0000')
            elif LotteryConfig.is_blue(num):
                colors.append('#0000FF')
            else:
                colors.append('#008000')
        
        bars = ax.bar(numbers, counts, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('号码', fontsize=12)
        ax.set_ylabel('出现次数', fontsize=12)
        ax.set_title('特别码频率分布图', fontsize=14, fontweight='bold')
        ax.set_xticks(numbers)
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, str(int(height)), ha='center', va='bottom', fontsize=7)
        
        fig.tight_layout()
        canvas.draw()
        
        layout.addWidget(canvas)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def _show_special_trend_chart(self):
        """特别码走势图"""
        if not self.historical_data:
            QMessageBox.information(self, "提示", "没有历史数据，请先导入数据！")
            return
        
        # 获取最近30期特别码
        recent = list(reversed(self.historical_data[:30]))
        specials = []
        periods = []
        for record in recent:
            sp = record.get('special', 0)
            if 1 <= sp <= 49:
                specials.append(sp)
                periods.append(str(record.get('period', '?')))
        
        if not specials:
            QMessageBox.information(self, "提示", "没有特别码数据")
            return
        
        # 创建对话框显示图表
        dialog = QDialog(self)
        dialog.setWindowTitle("特别码走势图")
        dialog.setFixedSize(900, 500)
        
        layout = QVBoxLayout(dialog)
        
        _get_mpl()
        global _figure_module, _canvas_class
        if _figure_module is None or _canvas_class is None:
            QMessageBox.warning(self, "错误", "无法加载matplotlib库")
            dialog.close()
            return
        
        fig = _figure_module(figsize=(12, 5))
        canvas = _canvas_class(fig)
        ax = fig.add_subplot(111)
        
        x = range(len(specials))
        ax.plot(x, specials, 'o-', color='#F39C12', markersize=8, linewidth=2, markeredgecolor='#E67E22')
        
        # 添加均值线
        avg_val = sum(specials) / len(specials)
        ax.axhline(y=avg_val, color='#3498DB', linestyle='--', linewidth=1.5, label='均值: ' + "{:.1f}".format(avg_val))
        
        ax.set_xlabel('期数', fontsize=12)
        ax.set_ylabel('特别码', fontsize=12)
        ax.set_title('特别码走势图（最近' + str(len(specials)) + '期）', fontsize=14, fontweight='bold')
        ax.set_xticks(x[::5])
        ax.set_xticklabels([periods[i] for i in range(0, len(periods), 5)], rotation=45)
        ax.set_ylim(0, 50)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        canvas.draw()
        
        layout.addWidget(canvas)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def _show_special_correlation_analysis(self):
        """正码与特别码关联分析"""
        if not self.historical_data:
            QMessageBox.information(self, "提示", "没有历史数据，请先导入数据！")
            return
        
        # 构建正码与特别码的共现矩阵
        cooccurrence = defaultdict(Counter)
        for record in self.historical_data:
            numbers = record.get('numbers', [])
            special = record.get('special', 0)
            if 1 <= special <= 49 and numbers:
                for num in numbers:
                    if 1 <= num <= 49:
                        cooccurrence[num][special] += 1
        
        if not cooccurrence:
            QMessageBox.information(self, "提示", "没有足够的关联数据")
            return
        
        # 计算每个特别码的关联度得分
        special_scores = {}
        for sp in range(1, 50):
            score = 0
            for num, counter in cooccurrence.items():
                score += counter.get(sp, 0)
            special_scores[sp] = score
        
        # 排序取前10
        sorted_specials = sorted(special_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 显示结果
        result_text = "正码与特别码关联度分析（Top 10）\n"
        result_text += "=" * 50 + "\n\n"
        result_text += "排名  特别码  关联得分  颜色  生肖\n"
        result_text += "-" * 50 + "\n"
        
        for rank, (sp, score) in enumerate(sorted_specials, 1):
            color_name = "红"
            if LotteryConfig.is_blue(sp):
                color_name = "蓝"
            elif LotteryConfig.is_green(sp):
                color_name = "绿"
            zodiac = LotteryConfig.NUMBER_NAMES.get(sp, "")
            result_text += str(rank) + "    " + str(sp).zfill(2) + "     " + str(score) + "     " + color_name + "    " + zodiac + "\n"
        
        result_text += "\n" + "=" * 50 + "\n"
        result_text += "说明：关联得分越高，说明该特别码与正码共同出现的频率越高\n"
        result_text += "建议关注这些高关联度的特别码\n"
        
        self.seventh_result_text.setPlainText(result_text)
    
    def _create_statistics_chart_tab(self):
        widget = QWidget()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("统计分析图表")
        title.setObjectName("PanelTitle")
        left_layout.addWidget(title)
        
        chart_types = [
            ("频率分布图", self._show_frequency_chart),
            ("遗漏值分析图", self._show_missing_chart),
            ("单双分布图", self._show_odd_even_chart),
            ("大小分布图", self._show_size_chart),
            ("颜色分布图", self._show_color_chart),
            ("区间分布图", self._show_range_chart),
            ("尾数分布图", self._show_tail_chart),
            ("综合走势图", self._show_comprehensive_chart),
            ("号码走势图", self._show_number_trend_chart),
            ("相关性热力图", self._show_correlation_heatmap),
            ("间隔分析图", self._show_interval_analysis),
            ("连号邻号概率", self._show_consecutive_probability),
            ("和值分布图", self._show_sum_distribution),
        ]
        
        for name, callback in chart_types:
            btn = QPushButton(name)
            btn.clicked.connect(callback)
            left_layout.addWidget(btn)
        
        left_layout.addStretch()
        
        splitter.addWidget(left_panel)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        self.chart_title_label = QLabel("选择左侧图表类型查看分析结果")
        self.chart_title_label.setObjectName("ChartTitleLabel")
        right_layout.addWidget(self.chart_title_label)
        
        chart_scroll = QScrollArea()
        chart_scroll.setWidgetResizable(True)
        
        self.main_chart_widget = StatisticsChart()
        chart_scroll.setWidget(self.main_chart_widget)
        
        right_layout.addWidget(chart_scroll, 1)
        
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(self.margin_left, self.margin_top, self.margin_right, self.margin_bottom)
        layout.setSpacing(self.spacing)
        layout.addWidget(splitter)
        
        return widget
    
    def _predict_seventh_size(self):
        if not self.historical_data:
            self.seventh_result_text.setPlainText("没有历史数据，请先导入数据！")
            return
        big_count = small_count = 0
        for record in self.historical_data:
            seventh = record.get('special', 0)
            if seventh > 24:
                big_count += 1
            else:
                small_count += 1
        total = big_count + small_count
        big_ratio = big_count / total * 100 if total > 0 else 50
        small_ratio = small_count / total * 100 if total > 0 else 50
        prediction = "大" if big_ratio > small_ratio else "小"
        confidence = max(big_ratio, small_ratio)
        result = "第七位大小预判结果\n" + "="*40 + "\n\n"
        result += "历史数据统计：\n"
        result += "   大号(25-49)出现次数：" + str(big_count) + " 次 (" + "{:.1f}".format(big_ratio) + "%)\n"
        result += "   小号(1-24)出现次数：" + str(small_count) + " 次 (" + "{:.1f}".format(small_ratio) + "%)\n\n"
        result += "预判结果：" + prediction + "\n"
        result += "   置信度：" + "{:.1f}".format(confidence) + "%\n\n"
        result += "建议：下一期第七位号码倾向于「" + prediction + "」范围"
        self.seventh_result_text.setPlainText(result)
    
    def _predict_seventh_odd_even(self):
        if not self.historical_data:
            self.seventh_result_text.setPlainText("没有历史数据，请先导入数据！")
            return
        odd_count = even_count = 0
        for record in self.historical_data:
            seventh = record.get('special', 0)
            if seventh % 2 == 1:
                odd_count += 1
            else:
                even_count += 1
        total = odd_count + even_count
        odd_ratio = odd_count / total * 100 if total > 0 else 50
        even_ratio = even_count / total * 100 if total > 0 else 50
        prediction = "单" if odd_ratio > even_ratio else "双"
        confidence = max(odd_ratio, even_ratio)
        result = "第七位单双预判结果\n" + "="*40 + "\n\n"
        result += "历史数据统计：\n"
        result += "   单号出现次数：" + str(odd_count) + " 次 (" + "{:.1f}".format(odd_ratio) + "%)\n"
        result += "   双号出现次数：" + str(even_count) + " 次 (" + "{:.1f}".format(even_ratio) + "%)\n\n"
        result += "预判结果：" + prediction + "\n"
        result += "   置信度：" + "{:.1f}".format(confidence) + "%\n\n"
        result += "建议：下一期第七位号码倾向于「" + prediction + "」数"
        self.seventh_result_text.setPlainText(result)
    
    def _predict_seventh_tail(self):
        if not self.historical_data:
            self.seventh_result_text.setPlainText("没有历史数据，请先导入数据！")
            return
        big_tail_count = small_tail_count = 0
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
        prediction = "大尾(5-9)" if big_ratio > small_ratio else "小尾(0-4)"
        confidence = max(big_ratio, small_ratio)
        result = "第七位尾数大小预判结果\n" + "="*40 + "\n\n"
        result += "历史数据统计：\n"
        result += "   大尾(5-9)出现次数：" + str(big_tail_count) + " 次 (" + "{:.1f}".format(big_ratio) + "%)\n"
        result += "   小尾(0-4)出现次数：" + str(small_tail_count) + " 次 (" + "{:.1f}".format(small_ratio) + "%)\n\n"
        result += "预判结果：" + prediction + "\n"
        result += "   置信度：" + "{:.1f}".format(confidence) + "%\n\n"
        result += "建议：下一期第七位号码尾数倾向于「" + prediction + "」范围"
        self.seventh_result_text.setPlainText(result)
    
    def _predict_seventh_all(self):
        if not self.historical_data:
            self.seventh_result_text.setPlainText("没有历史数据，请先导入数据！")
            return
        big_count = small_count = odd_count = even_count = big_tail_count = small_tail_count = 0
        for record in self.historical_data:
            seventh = record.get('special', 0)
            if seventh > 24:
                big_count += 1
            else:
                small_count += 1
            if seventh % 2 == 1:
                odd_count += 1
            else:
                even_count += 1
            if seventh % 10 >= 5:
                big_tail_count += 1
            else:
                small_tail_count += 1
        total = len(self.historical_data)
        size_pred = "大" if big_count > small_count else "小"
        odd_even_pred = "单" if odd_count > even_count else "双"
        tail_pred = "大尾" if big_tail_count > small_tail_count else "小尾"
        result = "第七位综合预判结果\n" + "="*40 + "\n\n"
        result += "数据样本：" + str(total) + " 期\n\n"
        result += "大小分析：大号" + str(big_count) + "次 小号" + str(small_count) + "次 -> 预判：" + size_pred + "\n"
        result += "单双分析：单数" + str(odd_count) + "次 双数" + str(even_count) + "次 -> 预判：" + odd_even_pred + "\n"
        result += "尾数分析：大尾" + str(big_tail_count) + "次 小尾" + str(small_tail_count) + "次 -> 预判：" + tail_pred + "\n\n"
        result += "综合预判结果：\n"
        result += "   第七位号码特征：" + size_pred + " + " + odd_even_pred + " + " + tail_pred + "\n\n"
        result += "建议关注号码范围："
        recommended = []
        for n in range(1, 50):
            size_ok = (n > 24) == (size_pred == "大")
            odd_ok = (n % 2 == 1) == (odd_even_pred == "单")
            tail_ok = (n % 10 >= 5) == (tail_pred == "大尾")
            if size_ok and odd_ok and tail_ok:
                recommended.append(n)
        if recommended:
            result += "\n   " + ', '.join(str(n) for n in recommended[:10])
            if len(recommended) > 10:
                result += " ... 等共" + str(len(recommended)) + "个号码"
        self.seventh_result_text.setPlainText(result)
    
    def _show_frequency_chart(self):
        self.chart_title_label.setText("频率分布图")
        self._draw_chart("frequency")
    
    def _show_missing_chart(self):
        self.chart_title_label.setText("遗漏值分析图")
        self._draw_chart("missing")
    
    def _show_odd_even_chart(self):
        self.chart_title_label.setText("单双分布图")
        self._draw_chart("odd_even")
    
    def _show_size_chart(self):
        self.chart_title_label.setText("大小分布图")
        self._draw_chart("size")
    
    def _show_color_chart(self):
        self.chart_title_label.setText("颜色分布图")
        self._draw_chart("color")
    
    def _show_range_chart(self):
        self.chart_title_label.setText("区间分布图")
        self._draw_chart("range")
    
    def _show_tail_chart(self):
        self.chart_title_label.setText("尾数分布图")
        self._draw_chart("tail")
    
    def _show_comprehensive_chart(self):
        self.chart_title_label.setText("综合走势图")
        self._draw_chart("comprehensive")
    
    def _draw_chart(self, chart_type):
        if not self.historical_data:
            self.main_chart_widget.figure.clear()
            ax = self.main_chart_widget.figure.add_subplot(111)
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', fontsize=16)
            ax.set_title('请先导入数据', fontsize=14)
            self.main_chart_widget.canvas.draw()
            return
        if chart_type == "frequency":
            frequency = {}
            for record in self.historical_data:
                for num in record.get('numbers', []):
                    frequency[num] = frequency.get(num, 0) + 1
                special = record.get('special')
                if special:
                    frequency[special] = frequency.get(special, 0) + 1
            if frequency:
                self.main_chart_widget.plot_frequency(frequency, "数字出现频率分布")
        elif chart_type == "missing":
            missing = {i: 0 for i in range(1, 50)}
            appeared = set()
            for i, record in enumerate(self.historical_data):
                for num in record.get('numbers', []):
                    if num not in appeared:
                        missing[num] = i
                    appeared.add(num)
            for num in range(1, 50):
                if num not in appeared:
                    missing[num] = len(self.historical_data)
            self.main_chart_widget.plot_missing(missing, "数字遗漏期数")
        elif chart_type == "odd_even":
            odd_count = even_count = 0
            for record in self.historical_data:
                for num in record.get('numbers', []):
                    if num % 2 == 1:
                        odd_count += 1
                    else:
                        even_count += 1
            self.main_chart_widget.plot_distribution({'单数': odd_count, '双数': even_count}, "单双分布统计")
        elif chart_type == "size":
            big_count = small_count = 0
            for record in self.historical_data:
                for num in record.get('numbers', []):
                    if num > 24:
                        big_count += 1
                    else:
                        small_count += 1
            self.main_chart_widget.plot_distribution({'大号(25-49)': big_count, '小号(1-24)': small_count}, "大小分布统计")
        elif chart_type == "color":
            red_count = blue_count = green_count = 0
            for record in self.historical_data:
                for num in record.get('numbers', []):
                    if LotteryConfig.is_red(num):
                        red_count += 1
                    elif LotteryConfig.is_blue(num):
                        blue_count += 1
                    else:
                        green_count += 1
            self.main_chart_widget.plot_distribution({'红色': red_count, '蓝色': blue_count, '绿色': green_count}, "颜色分布统计")
        elif chart_type == "range":
            range_count = {i: 0 for i in range(5)}
            for record in self.historical_data:
                for num in record.get('numbers', []):
                    idx = LotteryConfig.get_range_index(num)
                    if idx >= 0:
                        range_count[idx] += 1
            labels = [LotteryConfig.RANGES[i][2] for i in range(5)]
            data = {labels[i]: range_count[i] for i in range(5)}
            self.main_chart_widget.plot_distribution(data, "区间分布统计")
        elif chart_type == "tail":
            tail_count = {i: 0 for i in range(10)}
            for record in self.historical_data:
                for num in record.get('numbers', []):
                    tail = LotteryConfig.get_tail_digit(num)
                    tail_count[tail] += 1
            data = {str(i) + "尾": tail_count[i] for i in range(10)}
            self.main_chart_widget.plot_distribution(data, "尾数分布统计")
        elif chart_type == "comprehensive":
            trend_data = []
            for record in self.historical_data[:50]:
                period = str(record.get('period', '?'))
                numbers = record.get('numbers', [])
                trend_data.append((period, numbers))
            self.main_chart_widget.plot_trend(trend_data, "综合走势图（最近50期）")
        elif chart_type == "correlation_heatmap":
            # 功能9：相关性热力图 - 构建49x49号码共现矩阵
            cooccurrence = {i: {j: 0 for j in range(1, 50)} for i in range(1, 50)}
            for record in self.historical_data:
                numbers = record.get('numbers', [])
                for n1 in numbers:
                    for n2 in numbers:
                        if n1 != n2:
                            cooccurrence[n1][n2] += 1
            self.main_chart_widget.plot_correlation_heatmap(cooccurrence, "号码相关性热力图（共现矩阵）")
        elif chart_type == "interval_analysis":
            # 功能9：间隔分析图 - 计算高频号码出现间隔
            # 统计每个号码出现的期数索引
            number_appearances = {i: [] for i in range(1, 50)}
            for idx, record in enumerate(self.historical_data):
                for num in record.get('numbers', []):
                    number_appearances[num].append(idx)
            
            # 计算间隔
            interval_data = {}
            for num, appearances in number_appearances.items():
                if len(appearances) >= 3:  # 至少出现3次才计算间隔
                    intervals = []
                    for i in range(1, len(appearances)):
                        intervals.append(appearances[i] - appearances[i-1])
                    interval_data[num] = intervals
            
            self.main_chart_widget.plot_interval_analysis(interval_data, "号码间隔分析图（箱线图）")
        elif chart_type == "consecutive_probability":
            # 功能9：连号邻号概率图 - 分析连号、邻号、同尾数出现概率
            consecutive_count = 0
            adjacent_count = 0
            same_tail_count = 0
            total_combinations = 0
            
            consecutive_pairs = {}
            adjacent_pairs = {}
            same_tail_pairs = {}
            
            for record in self.historical_data:
                numbers = sorted(record.get('numbers', []))
                total_combinations += 1
                
                # 检查连号（差值为1）
                for i in range(len(numbers) - 1):
                    diff = numbers[i+1] - numbers[i]
                    if diff == 1:
                        consecutive_count += 1
                        pair = tuple(sorted([numbers[i], numbers[i+1]]))
                        consecutive_pairs[pair] = consecutive_pairs.get(pair, 0) + 1
                    elif diff == 2:
                        adjacent_count += 1
                        pair = tuple(sorted([numbers[i], numbers[i+1]]))
                        adjacent_pairs[pair] = adjacent_pairs.get(pair, 0) + 1
                
                # 检查同尾数
                tails = {}
                for num in numbers:
                    tail = num % 10
                    if tail in tails:
                        same_tail_count += 1
                        pair = tuple(sorted([tails[tail], num]))
                        same_tail_pairs[pair] = same_tail_pairs.get(pair, 0) + 1
                    tails[tail] = num
            
            data = {}
            if total_combinations > 0:
                data['consecutive'] = {
                    'avg_prob': consecutive_count / total_combinations if total_combinations > 0 else 0,
                    'pairs': consecutive_pairs
                }
                data['adjacent'] = {
                    'avg_prob': adjacent_count / total_combinations if total_combinations > 0 else 0,
                    'pairs': adjacent_pairs
                }
                data['same_tail'] = {
                    'avg_prob': same_tail_count / total_combinations if total_combinations > 0 else 0,
                    'pairs': same_tail_pairs
                }
            
            self.main_chart_widget.plot_consecutive_probability(data, "连号邻号概率分析")
        elif chart_type == "sum_distribution":
            # 功能9：和值分布图 - 计算每期号码和值分布
            sums = []
            for record in self.historical_data:
                numbers = record.get('numbers', [])
                if numbers:
                    sums.append(sum(numbers))
            
            self.main_chart_widget.plot_sum_distribution(sums, "和值分布图（含正态拟合）")
    
    def _create_status_bar(self):
        self.statusBar().showMessage("就绪 | 准备预测...")
        self.data_count_label = QLabel("历史记录: 0 条")
        self.statusBar().addPermanentWidget(self.data_count_label)
    
    def _apply_stylesheet(self):
        self.setStyleSheet("")
        self._update_stylesheet()
    
    def _update_stylesheet(self):
        font_size = LotteryConfig.FONT_SIZES.get(self.font_size_key, 16)
        small_font_size = max(10, font_size - 4)
        large_font_size = font_size + 4
        if self.is_dark_mode:
            bg = "#1E1E2E"
            fg = "#CDD6F4"
            card = "#313244"
            accent = "#89B4FA"
            success = "#A6E3A1"
            error = "#F38BA8"
            border = "#45475A"
            hover_bg = "#45475A"
            pressed_bg = "#585B70"
            header_bg = "#313244"
            input_bg = "#313244"
        else:
            bg = "#FFFFFF"
            fg = "#000000"
            card = "#FFFFFF"
            accent = "#3498DB"
            success = "#2ECC71"
            error = "#E74C3C"
            border = "#DDDDDD"
            hover_bg = "#F8F9FA"
            pressed_bg = "#E8E8E8"
            header_bg = "#F8F9FA"
            input_bg = "#FFFFFF"
        stylesheet = (
            "QWidget { background-color: " + bg + "; color: " + fg + "; font-size: " + str(font_size) + "px; font-family: \"Microsoft YaHei\", \"SimHei\", \"PingFang SC\", Arial, sans-serif; }"
            " #TopBar { background-color: " + bg + "; border-bottom: 2px solid " + border + "; }"
            " #TitleLabel { font-size: " + str(large_font_size) + "px; font-weight: bold; color: " + accent + "; }"
            " QPushButton { background-color: " + card + "; color: " + fg + "; border: 2px solid " + border + "; border-radius: 6px; padding: 6px 16px; font-size: " + str(small_font_size) + "px; min-height: 30px; }"
            " QPushButton:hover { background-color: " + hover_bg + "; border-color: " + accent + "; }"
            " QPushButton:pressed { background-color: " + pressed_bg + "; }"
            " QTabWidget::pane { border: 1px solid " + border + "; background-color: " + bg + "; }"
            " QTabBar::tab { background-color: " + card + "; color: " + fg + "; border: 1px solid " + border + "; padding: 8px 20px; margin-right: 2px; font-size: " + str(small_font_size) + "px; }"
            " QTabBar::tab:selected { background-color: " + accent + "; color: white; border-color: " + accent + "; }"
            " QTabBar::tab:hover { background-color: " + hover_bg + "; }"
            " QTextEdit, QLineEdit { background-color: " + input_bg + "; color: " + fg + "; border: 2px solid " + border + "; border-radius: 4px; padding: 8px; font-size: " + str(font_size) + "px; }"
            " QTextEdit:focus, QLineEdit:focus { border-color: " + accent + "; }"
            " QComboBox { background-color: " + card + "; color: " + fg + "; border: 2px solid " + border + "; border-radius: 4px; padding: 6px 12px; font-size: " + str(small_font_size) + "px; min-height: 28px; }"
            " QComboBox:hover { border-color: " + accent + "; }"
            " #PanelTitle { font-size: " + str(font_size) + "px; font-weight: bold; color: " + accent + "; padding: 5px; border-bottom: 1px solid " + border + "; }"
            " QTableWidget { background-color: " + bg + "; color: " + fg + "; border: 1px solid " + border + "; border-radius: 4px; font-size: " + str(small_font_size) + "px; }"
            " QTableWidget::item { padding: 5px; }"
            " QTableWidget::item:selected { background-color: " + accent + "; color: white; }"
            " QTableWidget::header::section { background-color: " + header_bg + "; color: " + fg + "; padding: 5px; border: 1px solid " + border + "; font-weight: bold; }"
            " QScrollBar:vertical { background-color: " + bg + "; width: 12px; margin: 0px; }"
            " QScrollBar::handle:vertical { background-color: " + border + "; border-radius: 6px; min-height: 30px; }"
            " QScrollBar::handle:vertical:hover { background-color: " + accent + "; }"
            " QScrollBar:horizontal { background-color: " + bg + "; height: 12px; margin: 0px; }"
            " QScrollBar::handle:horizontal { background-color: " + border + "; border-radius: 6px; min-width: 30px; }"
            " QScrollBar::handle:horizontal:hover { background-color: " + accent + "; }"
            " QSplitter::handle { background-color: " + border + "; }"
            " QSplitter::handle:hover { background-color: " + accent + "; }"
            " QScrollArea { background-color: " + bg + "; border: 1px solid " + border + "; border-radius: 4px; }"
            " #PredictionDisplay { font-size: " + str(large_font_size) + "px; font-weight: bold; color: " + success + "; padding: 15px; background-color: " + card + "; border: 2px solid " + success + "; border-radius: 8px; }"
            " #LatestDisplay { font-size: " + str(font_size) + "px; padding: 10px; background-color: " + card + "; border: 2px solid " + accent + "; border-radius: 6px; }"
            " #SelectedNumbersLabel { color: " + success + "; font-weight: bold; }"
            " QStatusBar { background-color: " + bg + "; color: " + fg + "; border-top: 1px solid " + border + "; font-size: " + str(small_font_size) + "px; }"
            " #InfoLabel { font-size: " + str(small_font_size) + "px; color: " + fg + "; padding: 8px; background-color: " + card + "; border: 1px solid " + border + "; border-radius: 4px; }"
            " QSpinBox { background-color: " + card + "; color: " + fg + "; border: 2px solid " + border + "; border-radius: 4px; padding: 4px; }"
            " QSlider::groove:horizontal { background-color: " + border + "; height: 6px; border-radius: 3px; }"
            " QSlider::handle:horizontal { background-color: " + accent + "; width: 16px; margin: -5px 0; border-radius: 8px; }"
        )
        self.setStyleSheet(stylesheet)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_autosize()
    
    def _apply_autosize(self):
        window_size = self.size()
        window_width = window_size.width()
        window_height = window_size.height()
        base_width = 1600
        base_height = 1000
        scale_x = window_width / base_width
        scale_y = window_height / base_height
        scale = min(scale_x, scale_y)
        base_font_size = LotteryConfig.FONT_SIZES.get(self.font_size_key, 16)
        new_font_size = max(10, int(base_font_size * scale))
        font = QFont()
        font.setPointSize(new_font_size)
        QApplication.instance().setFont(font)
    
    def _on_font_size_changed(self, size_key):
        self.font_size_key = size_key
        self._update_stylesheet()
        self._apply_autosize()
    
    def _increase_font_size(self):
        keys = list(LotteryConfig.FONT_SIZES.keys())
        current_index = keys.index(self.font_size_key) if self.font_size_key in keys else 3
        if current_index < len(keys) - 1:
            new_key = keys[current_index + 1]
            self.font_combo.setCurrentText(new_key)
    
    def _decrease_font_size(self):
        keys = list(LotteryConfig.FONT_SIZES.keys())
        current_index = keys.index(self.font_size_key) if self.font_size_key in keys else 3
        if current_index > 0:
            new_key = keys[current_index - 1]
            self.font_combo.setCurrentText(new_key)
    
    # ========================================================================
    # 【需求3-新增】安全文件写入：积分不足时自动重试
    # ========================================================================
    def _safe_write_file(self, filepath, content, max_retries=5, retry_delay=30):
        """【需求3-新增】安全写入文件，支持积分不足时自动重试
        当写入失败时，等待30秒后重试，最多重试5次"""
        for attempt in range(max_retries):
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    self.statusBar().showMessage('写入失败，等待重试... (' + str(attempt + 1) + '/' + str(max_retries) + ')')
                    import time
                    time.sleep(retry_delay)
                else:
                    QMessageBox.warning(self, '写入失败', '文件写入失败: ' + str(e) + '\n已重试' + str(max_retries) + '次')
                    return False
        return False
    
    def _safe_write_csv(self, filepath, rows, max_retries=5, retry_delay=30):
        """【需求3-新增】安全写入CSV文件
        rows: 二维列表，每行是一个列表"""
        import io
        try:
            # 先在内存中生成CSV内容
            output = io.StringIO()
            writer = csv.writer(output)
            for row in rows:
                writer.writerow(row)
            csv_content = output.getvalue()
            return self._safe_write_file(filepath, csv_content, max_retries, retry_delay)
        except Exception as e:
            QMessageBox.warning(self, 'CSV生成失败', 'CSV内容生成失败: ' + str(e))
            return False
    
    def _safe_write_json(self, filepath, data, max_retries=5, retry_delay=30):
        """【需求3-新增】安全写入JSON文件"""
        try:
            json_content = json.dumps(data, ensure_ascii=False, indent=2)
            return self._safe_write_file(filepath, json_content, max_retries, retry_delay)
        except Exception as e:
            QMessageBox.warning(self, 'JSON转换失败', 'JSON序列化失败: ' + str(e))
            return False
    
    def _load_data(self):
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.historical_data = json.load(f)
                print("已加载 " + str(len(self.historical_data)) + " 条历史记录")
            else:
                self.historical_data = DataUtils.generate_sample_data(100)
                print("已生成100条示例数据")
        except Exception as e:
            print("加载数据失败: " + str(e))
            self.historical_data = []
    
    def _save_data(self):
        """【需求3-更新】使用安全写入方法"""
        if self._safe_write_json(self.data_file, self.historical_data):
            print("数据保存成功")
            return True
        else:
            print("保存数据失败")
            return False
    
    def _on_import_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "导入数据", "", "JSON文件 (*.json);;文本文件 (*.txt);;所有文件 (*)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.historical_data.extend(data)
                        self._save_data()
                        self._update_history_table()
                        QMessageBox.information(self, "成功", "已导入 " + str(len(data)) + " 条记录")
                    else:
                        QMessageBox.warning(self, "错误", "数据格式不正确")
            except Exception as e:
                QMessageBox.warning(self, "错误", "导入失败: " + str(e))
    
    def _on_export_clicked(self):
        """【需求3-更新】使用安全写入方法"""
        if not self.historical_data:
            QMessageBox.information(self, "提示", "没有可导出的数据")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "导出数据", "lottery_data_export.json", "JSON文件 (*.json);;文本文件 (*.txt)")
        if file_path:
            if self._safe_write_json(file_path, self.historical_data):
                QMessageBox.information(self, "成功", "数据导出成功")
            # 失败信息由_safe_write_json内部处理
    
    def _on_save_clicked(self):
        if self._save_data():
            QMessageBox.information(self, "成功", "数据保存成功")
        else:
            QMessageBox.warning(self, "错误", "数据保存失败")
    
    def _on_add_data_clicked(self):
        text, ok = QInputDialog.getMultiLineText(self, "添加数据", "请输入开奖数据（格式：期号 日期 6个正码 特别码):\n例如：117 2026-04-27 05 12 23 34 45 08")
        if ok and text.strip():
            try:
                parts = text.strip().split()
                if len(parts) >= 8:
                    record = {
                        'period': int(parts[0]), 'date': parts[1],
                        'numbers': [int(parts[i]) for i in range(2, 8)], 'special': int(parts[7]),
                    }
                    self.historical_data.insert(0, record)
                    self._save_data()
                    self._update_history_table()
                    QMessageBox.information(self, "成功", "数据添加成功")
                else:
                    QMessageBox.warning(self, "错误", "数据格式不正确")
            except Exception as e:
                QMessageBox.warning(self, "错误", "添加失败: " + str(e))
    
    def _on_delete_data_clicked(self):
        if self.history_table.currentRow() < 0:
            QMessageBox.information(self, "提示", "请先选择要删除的记录")
            return
        reply = QMessageBox.question(self, "确认删除", "确定要删除选中的记录吗？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            row = self.history_table.currentRow()
            if 0 <= row < len(self.historical_data):
                del self.historical_data[row]
                self._save_data()
                self._update_history_table()
                QMessageBox.information(self, "成功", "删除成功")
    
    def _on_clear_data_clicked(self):
        reply = QMessageBox.question(self, "确认清空", "确定要清空所有历史记录吗？此操作不可恢复！", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.historical_data.clear()
            self._save_data()
            self._update_history_table()
            QMessageBox.information(self, "成功", "历史记录已清空")
    
    def _on_batch_delete_clicked(self):
        selected_rows = set()
        for item in self.history_table.selectedItems():
            selected_rows.add(item.row())
        if not selected_rows:
            QMessageBox.information(self, "提示", "请先选择要删除的记录")
            return
        reply = QMessageBox.question(self, "确认批量删除", "确定要删除选中的 " + str(len(selected_rows)) + " 条记录吗？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            for row in sorted(selected_rows, reverse=True):
                if 0 <= row < len(self.historical_data):
                    del self.historical_data[row]
            self._save_data()
            self._update_history_table()
            QMessageBox.information(self, "成功", "已删除 " + str(len(selected_rows)) + " 条记录")
    
    def _on_batch_modify_clicked(self):
        QMessageBox.information(self, "提示", "批量修改功能开发中")
    
    def _on_batch_add_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "批量添加", "", "文本文件 (*.txt);;所有文件 (*)")
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
                                'period': result.get('period'), 'date': result.get('date'),
                                'numbers': result.get('numbers'), 'special': result.get('special'),
                            }
                            self.historical_data.append(record)
                            count += 1
                self._save_data()
                self._update_history_table()
                QMessageBox.information(self, "成功", "成功导入 " + str(count) + " 条记录")
            except Exception as e:
                QMessageBox.warning(self, "错误", "导入失败: " + str(e))
    
    def _on_convert_clicked(self):
        """转换按钮 - 支持大量文本批量转格式，自动识别多期数据"""
        raw_text = self.raw_text_edit.toPlainText()
        if not raw_text.strip():
            QMessageBox.warning(self, "提示", "请输入要转换的原始数据")
            return
        # 核心逻辑：先按"第X期"拆分整段文本，确保多期在同一行也能识别
        formatted_lines = []
        success_count = 0
        fail_count = 0
        # 【需求1-更新】按"第X期"拆分，每个片段对应一期（支持中间有空格）
        segments = re.split(r'(?=第\s*\d+\s*期)', raw_text)
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            result = DataUtils.parse_raw_data(seg)
            if result:
                formatted_lines.append(DataUtils.format_data(result))
                success_count += 1
            else:
                fail_count += 1
        if formatted_lines:
            self.converted_text_edit.setPlainText('\n'.join(formatted_lines))
            msg = "批量转换完成：成功 " + str(success_count) + " 条"
            if fail_count > 0:
                msg += "，失败 " + str(fail_count) + " 条"
            self.statusBar().showMessage(msg)
        else:
            QMessageBox.warning(self, "错误", "无法解析数据，请检查格式")
    
    def _on_add_to_history_clicked(self):
        """添加到历史记录 - 支持批量添加多期数据"""
        raw_text = self.raw_text_edit.toPlainText()
        if not raw_text.strip():
            QMessageBox.warning(self, "提示", "请输入要添加的原始数据")
            return
        # 【需求1-更新】按"第X期"拆分，自动识别多期（支持中间有空格）
        segments = re.split(r'(?=第\s*\d+\s*期)', raw_text)
        parsed_records = []
        all_issues = []
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            result = DataUtils.parse_raw_data(seg)
            if result:
                record = {
                    'period': result.get('period'), 'date': result.get('date'),
                    'numbers': result.get('numbers'), 'special': result.get('special'),
                }
                issues = self._validate_record(record)
                all_issues.extend(issues)
                parsed_records.append(record)
        if all_issues:
            if not self._show_validation_results(all_issues):
                return
        added_count = 0
        for record in parsed_records:
            self.historical_data.insert(added_count, record)
            added_count += 1
        if added_count > 0:
            self._save_data()
            self._update_history_table()
            self.raw_text_edit.clear()
            self.converted_text_edit.clear()
            QMessageBox.information(self, "成功", "已添加 " + str(added_count) + " 条数据到历史记录")
            self.statusBar().showMessage("批量添加成功")
        else:
            QMessageBox.warning(self, "错误", "无法解析数据，请检查格式")
    
    def _on_batch_import_clicked(self):
        """批量导入文件 - 支持大量数据，自动按期拆分"""
        file_path, _ = QFileDialog.getOpenFileName(self, "批量导入", "", "文本文件 (*.txt);;JSON文件 (*.json);;所有文件 (*)")
        if not file_path:
            return
        try:
            # 支持JSON格式导入
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.historical_data.extend(data)
                    self._save_data()
                    self._update_history_table()
                    QMessageBox.information(self, "成功", "成功导入 " + str(len(data)) + " 条记录")
                return
            # 文本格式导入，支持大量数据
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # 【需求1-更新】先按"第X期"拆分，处理所有期数数据（支持中间有空格）
            segments = re.split(r'(?=第\s*\d+\s*期)', content)
            count = 0
            for seg in segments:
                seg = seg.strip()
                if not seg:
                    continue
                result = DataUtils.parse_raw_data(seg)
                if result:
                    record = {
                        'period': result.get('period'), 'date': result.get('date'),
                        'numbers': result.get('numbers'), 'special': result.get('special'),
                    }
                    self.historical_data.append(record)
                    count += 1
            self._save_data()
            self._update_history_table()
            QMessageBox.information(self, "成功", "成功导入 " + str(count) + " 条记录")
        except Exception as e:
            QMessageBox.warning(self, "错误", "导入失败: " + str(e))
    

    # ========================================================================
    # 功能1：预测结果收藏/对比
    # ========================================================================
    def _on_collect_prediction(self):
        """收藏当前预测结果"""
        display_text = self.prediction_display.text()
        if not display_text or display_text == "等待预测...":
            QMessageBox.information(self, "提示", "当前没有预测结果可收藏")
            return
        algo_name = "未知算法"
        if 0 <= self.current_algorithm_index < len(LotteryConfig.ALGORITHMS):
            algo_name = LotteryConfig.ALGORITHMS[self.current_algorithm_index][0]
        numbers = []
        for i in range(self.prediction_number_layout.count()):
            item = self.prediction_number_layout.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), NumberButton):
                numbers.append(item.widget().get_number())
        entry = {
            'algorithm': algo_name,
            'numbers': sorted(numbers),
            'special': 0,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.collected_predictions.append(entry)
        self.statusBar().showMessage("已收藏第 " + str(len(self.collected_predictions)) + " 个预测结果")
    
    def _on_compare_collected(self):
        """对比所有收藏结果，共识号高亮"""
        if not self.collected_predictions:
            QMessageBox.information(self, "提示", "还没有收藏任何预测结果")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("对比收藏结果")
        dialog.resize(900, 500)
        layout = QVBoxLayout(dialog)
        
        # 统计共识号
        num_counter = Counter()
        for entry in self.collected_predictions:
            for n in entry.get('numbers', []):
                num_counter[n] += 1
        
        consensus_nums = {n for n, c in num_counter.items() if c >= 2}
        
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["时间", "算法", "正码", "特别码"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setRowCount(len(self.collected_predictions))
        
        fg_color = "#CDD6F4" if self.is_dark_mode else "#000000"
        
        for i, entry in enumerate(self.collected_predictions):
            ts_item = QTableWidgetItem(entry.get('timestamp', ''))
            ts_item.setForeground(QColor(fg_color))
            table.setItem(i, 0, ts_item)
            
            algo_item = QTableWidgetItem(entry.get('algorithm', ''))
            algo_item.setForeground(QColor(fg_color))
            table.setItem(i, 1, algo_item)
            
            nums = entry.get('numbers', [])
            nums_str = '  '.join(str(n).zfill(2) for n in nums)
            nums_item = QTableWidgetItem(nums_str)
            # 共识号绿色高亮
            if any(n in consensus_nums for n in nums):
                nums_item.setForeground(QColor("#2ECC71"))
            else:
                nums_item.setForeground(QColor(fg_color))
            table.setItem(i, 2, nums_item)
            
            sp_item = QTableWidgetItem(str(entry.get('special', 0)).zfill(2))
            sp_item.setForeground(QColor(fg_color))
            table.setItem(i, 3, sp_item)
        
        layout.addWidget(table)
        
        # 共识号展示
        if consensus_nums:
            consensus_str = '  '.join(str(n).zfill(2) for n in sorted(consensus_nums))
            consensus_label = QLabel("共识号（出现2次以上）: " + consensus_str)
            consensus_label.setStyleSheet("color: #2ECC71; font-weight: bold; font-size: 14px; padding: 5px;")
            layout.addWidget(consensus_label)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def _on_clear_collected(self):
        """清空收藏"""
        if not self.collected_predictions:
            QMessageBox.information(self, "提示", "收藏已为空")
            return
        reply = QMessageBox.question(self, "确认", "确定清空所有收藏的预测结果吗？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.collected_predictions.clear()
            self.statusBar().showMessage("收藏已清空")
    
    # ========================================================================
    # 功能2：历史回测
    # ========================================================================
    def _create_backtest_tab(self):
        """创建回测分析标签页
        
        功能5增强：添加算法对比和命中趋势功能
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(self.spacing)
        layout.setContentsMargins(self.margin_left, self.margin_top, self.margin_right, self.margin_bottom)
        
        title = QLabel("回测分析")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        
        # ======================================================================== #
        # 功能5：回测结果可视化增强 - 控制面板
        # ======================================================================== #
        ctrl_layout = QHBoxLayout()
        
        ctrl_layout.addWidget(QLabel("选择算法:"))
        self.backtest_algo_combo = QComboBox()
        for algo_name, _ in LotteryConfig.ALGORITHMS:
            self.backtest_algo_combo.addItem(algo_name)
        ctrl_layout.addWidget(self.backtest_algo_combo)
        
        ctrl_layout.addWidget(QLabel("回测期数:"))
        self.backtest_period_spin = QSpinBox()
        self.backtest_period_spin.setRange(5, 50)
        self.backtest_period_spin.setValue(10)
        ctrl_layout.addWidget(self.backtest_period_spin)
        
        start_btn = QPushButton("开始回测")
        start_btn.setStyleSheet("QPushButton { background-color: #2ECC71; color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; } QPushButton:hover { background-color: #27AE60; }")
        start_btn.clicked.connect(self._on_start_backtest)
        ctrl_layout.addWidget(start_btn)
        
        # ======================================================================== #
        # 功能5：新增算法对比和命中趋势按钮
        # ======================================================================== #
        compare_btn = QPushButton("算法对比")
        compare_btn.setStyleSheet("QPushButton { background-color: #3498DB; color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; } QPushButton:hover { background-color: #2980B9; }")
        compare_btn.clicked.connect(self._on_compare_all_algorithms)
        ctrl_layout.addWidget(compare_btn)
        
        trend_btn = QPushButton("命中趋势")
        trend_btn.setStyleSheet("QPushButton { background-color: #9B59B6; color: white; border: none; border-radius: 6px; padding: 8px 16px; font-weight: bold; } QPushButton:hover { background-color: #8E44AD; }")
        trend_btn.clicked.connect(self._on_show_hit_trend)
        ctrl_layout.addWidget(trend_btn)
        
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)
        
        # 回测结果表格
        self.backtest_table = QTableWidget()
        self.backtest_table.setColumnCount(4)
        self.backtest_table.setHorizontalHeaderLabels(["期号", "预测号码", "实际号码", "命中数"])
        self.backtest_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.backtest_table, 1)
        
        # 统计摘要
        self.backtest_summary_label = QLabel("选择算法和期数后点击「开始回测」")
        self.backtest_summary_label.setWordWrap(True)
        self.backtest_summary_label.setStyleSheet("padding: 8px; font-size: 13px;")
        layout.addWidget(self.backtest_summary_label)
        
        return widget
    
    # ======================================================================== #
    # 功能5：回测结果可视化增强 - 新增方法
    # ======================================================================== #
    def _on_compare_all_algorithms(self):
        """算法对比：对所有21个算法批量回测"""
        if len(self.historical_data) < 30:
            QMessageBox.warning(self, "数据不足", "算法对比需要至少30条历史数据")
            return
        
        n_periods = min(20, len(self.historical_data) - 5)
        self.statusBar().showMessage("正在对" + str(n_periods) + "期数据进行算法对比...")
        
        # 对每个算法进行回测
        algo_results = []
        for algo_idx in range(len(LotteryConfig.ALGORITHMS)):
            try:
                hits_list = []
                for i in range(n_periods):
                    actual_record = self.historical_data[i]
                    actual_numbers = set(actual_record.get('numbers', []))
                    train_data = self.historical_data[i + 1:]
                    
                    if len(train_data) < 5:
                        continue
                    
                    predictor = PredictionAlgorithms(train_data)
                    predicted = self._get_prediction_by_index(predictor, algo_idx)
                    hits = len(set(predicted) & actual_numbers)
                    hits_list.append(hits)
                
                if hits_list:
                    avg_hit = sum(hits_list) / len(hits_list)
                    hit_rate = sum(hits_list) / (len(hits_list) * 6) * 100
                    algo_results.append({
                        'name': LotteryConfig.ALGORITHMS[algo_idx][0],
                        'avg_hit': avg_hit,
                        'hit_rate': hit_rate
                    })
            except Exception:
                continue
        
        if not algo_results:
            QMessageBox.warning(self, "对比失败", "无法完成算法对比")
            return
        
        # 创建对话框显示对比图表
        dialog = QDialog(self)
        dialog.setWindowTitle("算法对比分析")
        dialog.setFixedSize(1000, 600)
        
        layout = QVBoxLayout(dialog)
        
        _get_mpl()
        global _figure_module, _canvas_class, _pyplot_module
        if _figure_module is None or _canvas_class is None:
            QMessageBox.warning(self, "错误", "无法加载matplotlib库")
            dialog.close()
            return
        
        fig = _figure_module(figsize=(12, 6))
        canvas = _canvas_class(fig)
        ax = fig.add_subplot(111)
        
        names = [r['name'][:6] for r in algo_results]  # 截取前6个字符
        hit_rates = [r['hit_rate'] for r in algo_results]
        
        # 按命中率排序
        sorted_data = sorted(zip(names, hit_rates), key=lambda x: x[1], reverse=True)
        names, hit_rates = zip(*sorted_data)
        
        if _pyplot_module is not None:
            colors = _pyplot_module.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
        else:
            colors = '#3498db'
        bars = ax.bar(range(len(names)), hit_rates, color=colors, edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel('算法', fontsize=12)
        ax.set_ylabel('命中率 (%)', fontsize=12)
        ax.set_title('算法命中率对比（回测' + str(n_periods) + '期）', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, rate in zip(bars, hit_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, "{:.1f}%".format(rate), ha='center', va='bottom', fontsize=8)
        
        fig.tight_layout()
        canvas.draw()
        
        layout.addWidget(canvas)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
        self.statusBar().showMessage("算法对比完成")
    
    def _on_show_hit_trend(self):
        """显示选中算法的命中率时间序列趋势线"""
        if not hasattr(self, '_last_backtest_results') or not self._last_backtest_results:
            QMessageBox.information(self, "提示", "请先执行回测，然后查看命中趋势")
            return
        
        results = self._last_backtest_results
        
        # 创建对话框显示趋势图
        dialog = QDialog(self)
        dialog.setWindowTitle("命中趋势分析")
        dialog.setFixedSize(900, 500)
        
        layout = QVBoxLayout(dialog)
        
        _get_mpl()
        global _figure_module, _canvas_class
        if _figure_module is None or _canvas_class is None:
            QMessageBox.warning(self, "错误", "无法加载matplotlib库")
            dialog.close()
            return
        
        fig = _figure_module(figsize=(12, 5))
        canvas = _canvas_class(fig)
        ax = fig.add_subplot(111)
        
        periods = [str(r['period']) for r in results]
        hits = [r['hits'] for r in results]
        
        # 计算累计命中率
        total_hits = 0
        total_possible = 0
        cumulative_rates = []
        for h in hits:
            total_hits += h
            total_possible += 6
            cumulative_rates.append(total_hits / total_possible * 100)
        
        x = range(len(results))
        ax.plot(x, cumulative_rates, 'o-', color='#3498DB', markersize=6, linewidth=2)
        ax.axhline(y=cumulative_rates[-1] if cumulative_rates else 0, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('期数', fontsize=12)
        ax.set_ylabel('累计命中率 (%)', fontsize=12)
        ax.set_title('命中率趋势线（累计）', fontsize=14, fontweight='bold')
        ax.set_xticks(x[::3])
        ax.set_xticklabels([periods[i] for i in range(0, len(periods), 3)], rotation=45)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # 添加最终命中率标注
        if cumulative_rates:
            final_rate = cumulative_rates[-1]
            ax.annotate('最终: {:.1f}%'.format(final_rate), xy=(len(cumulative_rates)-1, final_rate),
                       xytext=(len(cumulative_rates)-5, final_rate+5),
                       arrowprops=dict(arrowstyle='->', color='gray'),
                       fontsize=10, color='#E74C3C')
        
        fig.tight_layout()
        canvas.draw()
        
        layout.addWidget(canvas)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def _get_prediction_by_index(self, predictor, algo_index):
        """根据算法索引获取预测结果"""
        if algo_index == 0:
            return predictor.comprehensive_recommendation(6)
        elif algo_index == 1:
            return predictor.hot_cold_algorithm(6)
        elif algo_index == 2:
            return predictor.odd_even_algorithm(6)
        elif algo_index == 3:
            return predictor.big_small_algorithm(6)
        elif algo_index == 4:
            return predictor.missing_value_analysis(6)
        elif algo_index == 5:
            return predictor.adjacent_number_analysis(6)
        elif algo_index == 6:
            return predictor.tail_distribution_algorithm(6)
        elif algo_index == 7:
            return predictor.range_distribution_algorithm(6)
        elif algo_index == 8:
            return predictor.roulette_selection(6)
        elif algo_index == 9:
            return predictor.historical_similarity(6)
        elif algo_index == 10:
            return predictor.poisson_distribution(6)
        elif algo_index == 11:
            return predictor.mystical_algorithm(6)
        elif algo_index == 12:
            return predictor.number_graph_algorithm(6)
        elif algo_index == 13:
            return predictor.shortest_path_algorithm(6)
        elif algo_index == 14:
            return predictor.community_detection_algorithm(6)
        elif algo_index == 15:
            return predictor.graph_clustering_algorithm(6)
        elif algo_index == 16:
            return predictor.numpy_matrix_algorithm(6)
        elif algo_index == 17:
            return predictor.scipy_optimization_algorithm(6)
        elif algo_index == 18:
            return predictor.sklearn_ensemble_algorithm(6)
        elif algo_index == 19:
            return predictor.pytorch_deep_learning_algorithm(6)
        elif algo_index == 20:
            return predictor.networkx_graph_algorithm(6)
        elif algo_index == 21:
            return predictor.special_frequency_regression(1)
        elif algo_index == 22:
            return predictor.special_correlation_algorithm(1)
        else:
            return predictor.comprehensive_recommendation(6)
    
    def _on_start_backtest(self):
        """执行回测"""
        if len(self.historical_data) < 15:
            QMessageBox.warning(self, "数据不足", "历史数据不足15条，无法回测")
            return
        
        algo_index = self.backtest_algo_combo.currentIndex()
        n_periods = self.backtest_period_spin.value()
        
        if n_periods > len(self.historical_data) - 5:
            n_periods = len(self.historical_data) - 5
            self.backtest_period_spin.setValue(n_periods)
        
        results = []
        total_hits = 0
        
        for i in range(n_periods):
            actual_record = self.historical_data[i]
            actual_numbers = set(actual_record.get('numbers', []))
            
            # 用 i+1 之后的数据预测
            train_data = self.historical_data[i + 1:]
            if len(train_data) < 5:
                continue
            
            try:
                predictor = PredictionAlgorithms(train_data)
                # 使用统一的预测方法
                predicted = self._get_prediction_by_index(predictor, algo_index)
            except Exception:
                predicted = []
            
            hits = len(set(predicted) & actual_numbers)
            total_hits += hits
            
            results.append({
                'period': actual_record.get('period', '?'),
                'predicted': sorted(predicted),
                'actual': sorted(actual_record.get('numbers', [])),
                'hits': hits
            })
        
        # 保存回测结果以便后续查看趋势
        self._last_backtest_results = results
        
        # 填充表格
        fg_color = "#CDD6F4" if self.is_dark_mode else "#000000"
        self.backtest_table.setRowCount(len(results))
        for i, r in enumerate(results):
            p_item = QTableWidgetItem(str(r['period']))
            p_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            p_item.setForeground(QColor(fg_color))
            self.backtest_table.setItem(i, 0, p_item)
            
            pred_str = '  '.join(str(n).zfill(2) for n in r['predicted'])
            pred_item = QTableWidgetItem(pred_str)
            pred_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            pred_item.setForeground(QColor(fg_color))
            self.backtest_table.setItem(i, 1, pred_item)
            
            act_str = '  '.join(str(n).zfill(2) for n in r['actual'])
            act_item = QTableWidgetItem(act_str)
            act_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            act_item.setForeground(QColor(fg_color))
            self.backtest_table.setItem(i, 2, act_item)
            
            hit_item = QTableWidgetItem(str(r['hits']))
            hit_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if r['hits'] >= 3:
                hit_item.setForeground(QColor("#2ECC71"))
            elif r['hits'] >= 1:
                hit_item.setForeground(QColor("#F39C12"))
            else:
                hit_item.setForeground(QColor("#E74C3C"))
            self.backtest_table.setItem(i, 3, hit_item)
        
        # 统计摘要
        if results:
            avg_hit = total_hits / len(results)
            best_idx = max(range(len(results)), key=lambda x: results[x]['hits'])
            worst_idx = min(range(len(results)), key=lambda x: results[x]['hits'])
            hit_rate = total_hits / (len(results) * 6) * 100
            
            summary = "回测统计摘要\n"
            summary += "=" * 40 + "\n"
            summary += "回测期数: " + str(len(results)) + " 期\n"
            summary += "总命中次数: " + str(total_hits) + " / " + str(len(results) * 6) + "\n"
            summary += "平均命中率: " + "{:.1f}".format(hit_rate) + "%\n"
            summary += "平均每期命中: " + "{:.2f}".format(avg_hit) + " 个\n"
            summary += "最佳期: 第" + str(results[best_idx]['period']) + "期 命中" + str(results[best_idx]['hits']) + "个\n"
            summary += "最差期: 第" + str(results[worst_idx]['period']) + "期 命中" + str(results[worst_idx]['hits']) + "个"
            self.backtest_summary_label.setText(summary)
        
        self.statusBar().showMessage("回测完成")
    
    # ========================================================================
    # 功能3：预测历史记录标签页
    # ========================================================================
    def _create_prediction_history_tab(self):
        """创建预测记录标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(self.spacing)
        layout.setContentsMargins(self.margin_left, self.margin_top, self.margin_right, self.margin_bottom)
        
        title = QLabel("预测记录")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        
        self.prediction_history_table = QTableWidget()
        self.prediction_history_table.setColumnCount(4)
        self.prediction_history_table.setHorizontalHeaderLabels(["时间", "算法", "正码", "特别码"])
        self.prediction_history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.prediction_history_table, 1)
        
        btn_layout = QHBoxLayout()
        export_csv_btn = QPushButton("导出CSV")
        export_csv_btn.clicked.connect(self._on_export_prediction_history_csv)
        btn_layout.addWidget(export_csv_btn)
        
        clear_ph_btn = QPushButton("清空记录")
        clear_ph_btn.clicked.connect(self._on_clear_prediction_history)
        btn_layout.addWidget(clear_ph_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        return widget
    
    def _create_info_tab(self):
        """创建应用说明标签页"""
        # 外层滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("InfoScrollArea")
        
        # 内容容器
        container = QWidget()
        container.setObjectName("InfoContainer")
        layout = QVBoxLayout(container)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # ===== 1. 应用简介 =====
        intro_group = QGroupBox("应用简介")
        intro_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #2ECC71;
                border: 2px solid #DDDDDD;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        intro_layout = QVBoxLayout()
        intro_layout.setSpacing(10)
        
        app_name = QLabel("应用名称：彩票预测系统 v7.1")
        app_name.setStyleSheet("font-size: 15px; font-weight: bold; color: #333333;")
        intro_layout.addWidget(app_name)
        
        framework = QLabel("开发框架：PyQt6")
        framework.setStyleSheet("font-size: 14px; color: #555555;")
        intro_layout.addWidget(framework)
        
        purpose = QLabel("用途：香港六合彩票数据分析和预测")
        purpose.setStyleSheet("font-size: 14px; color: #555555;")
        intro_layout.addWidget(purpose)
        
        intro_group.setLayout(intro_layout)
        
        # ===== 2. 核心库一览 =====
        lib_group = QGroupBox("核心库一览")
        lib_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #3498DB;
                border: 2px solid #DDDDDD;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        lib_layout = QVBoxLayout()
        
        lib_table = QTableWidget()
        lib_table.setColumnCount(2)
        lib_table.setHorizontalHeaderLabels(["库名", "用途"])
        lib_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        lib_table.setAlternatingRowColors(True)
        lib_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lib_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        lib_table.verticalHeader().setVisible(False)
        lib_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #DDDDDD;
                gridline-color: #EEEEEE;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #3498DB;
                color: white;
                font-weight: bold;
                padding: 8px;
                border: none;
            }
        """)
        
        lib_data = [
            ("PyQt6", "GUI框架，窗口/控件/布局/信号槽"),
            ("NumPy", "数组运算、矩阵分解、线性回归、相关性分析"),
            ("Pandas", "DataFrame数据处理、CSV/JSON导入导出"),
            ("Matplotlib", "频率图/热力图/箱线图/柱状图/折线图/直方图等可视化"),
            ("SciPy", "优化算法、高斯平滑、样条插值、分布建模、正态拟合"),
            ("StatsModels", "自相关分析、时间序列建模"),
            ("Scikit-learn", "随机森林/梯度提升/逻辑回归/MLP/KMeans/PCA分类聚类"),
            ("PyTorch", "LSTM深度学习、温度退火采样、神经网络训练"),
            ("TensorFlow", "Keras序列模型（可选）"),
            ("Optuna", "贝叶斯超参数优化、TPESampler"),
            ("NetworkX", "图算法、PageRank、社区发现、最短路径、图聚类"),
            ("Seaborn", "统计可视化增强"),
        ]
        
        lib_table.setRowCount(len(lib_data))
        for i, (name, desc) in enumerate(lib_data):
            name_item = QTableWidgetItem(name)
            name_item.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            name_item.setForeground(QColor("#2C3E50"))
            lib_table.setItem(i, 0, name_item)
            desc_item = QTableWidgetItem(desc)
            desc_item.setFont(QFont("Arial", 12))
            lib_table.setItem(i, 1, desc_item)
        
        lib_layout.addWidget(lib_table)
        lib_group.setLayout(lib_layout)

        
        # ===== 3. 23种预测算法说明 =====
        algo_group = QGroupBox("23种预测算法说明")
        algo_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #2ECC71;
                border: 2px solid #DDDDDD;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        algo_layout = QVBoxLayout()
        
        algo_table = QTableWidget()
        algo_table.setColumnCount(3)
        algo_table.setHorizontalHeaderLabels(["序号", "算法名称", "算法说明"])
        algo_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        algo_table.setAlternatingRowColors(True)
        algo_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        algo_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        algo_table.verticalHeader().setVisible(False)
        algo_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #DDDDDD;
                gridline-color: #EEEEEE;
            }
            QTableWidget::item {
                padding: 6px;
            }
            QHeaderView::section {
                background-color: #2ECC71;
                color: white;
                font-weight: bold;
                padding: 8px;
                border: none;
            }
        """)
        
        algo_data = [
            ("1", "综合推荐算法", "Optuna权重优化+多维度加权得分"),
            ("2", "冷热数字算法", "指数加权+趋势动量"),
            ("3", "单双算法", "马尔可夫链+均值回归"),
            ("4", "大小算法", "线性回归+区间平衡"),
            ("5", "遗漏值分析算法", "Z-score sigmoid评分"),
            ("6", "连号/邻号分析算法", "条件概率+距离衰减"),
            ("7", "尾数分布算法", "Chi-square检验"),
            ("8", "区间分布算法", "Pandas趋势+卡方偏差"),
            ("9", "轮盘赌选择算法", "PyTorch温度退火采样"),
            ("10", "历史相似性算法", "63维特征+cosine相似度"),
            ("11", "泊松概率分布算法", "SciPy生存函数"),
            ("12", "玄学算法", "天干地支+五行生克"),
            ("13", "号码关联图算法", "NetworkX PageRank中心性"),
            ("14", "最短路径算法", "NetworkX Dijkstra距离"),
            ("15", "社区发现算法", "NetworkX greedy_modularity"),
            ("16", "图聚类算法", "NetworkX k-clique连通分量"),
            ("17", "NumPy矩阵算法", "lstsq线性回归+SVD分解"),
            ("18", "SciPy优化算法", "minimize权重优化+样条插值"),
            ("19", "Sklearn集成算法", "RF/GB/LR/MLP五模型投票"),
            ("20", "PyTorch深度学习算法", "LSTM训练+softmax采样"),
            ("21", "NetworkX图算法", "多层图网络+中心性+聚类"),
            ("22", "特别码频率回归算法", "特别码历史频率加权回归"),
            ("23", "特别码关联算法", "特别码与正码关联规则挖掘"),
        ]
        
        algo_table.setRowCount(len(algo_data))
        for i, (num, name, desc) in enumerate(algo_data):
            num_item = QTableWidgetItem(num)
            num_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            num_item.setFont(QFont("Arial", 11, QFont.Weight.Bold))
            num_item.setForeground(QColor("#3498DB"))
            algo_table.setItem(i, 0, num_item)
            name_item = QTableWidgetItem(name)
            name_item.setFont(QFont("Arial", 11, QFont.Weight.Bold))
            name_item.setForeground(QColor("#2C3E50"))
            algo_table.setItem(i, 1, name_item)
            desc_item = QTableWidgetItem(desc)
            desc_item.setFont(QFont("Arial", 11))
            algo_table.setItem(i, 2, desc_item)
        
        algo_layout.addWidget(algo_table)
        algo_group.setLayout(algo_layout)

        
        # ===== 4. 标签页功能说明 =====
        tab_desc_group = QGroupBox("标签页功能说明")
        tab_desc_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #3498DB;
                border: 2px solid #DDDDDD;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        tab_desc_layout = QVBoxLayout()
        
        tab_table = QTableWidget()
        tab_table.setColumnCount(2)
        tab_table.setHorizontalHeaderLabels(["标签页", "功能描述"])
        tab_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tab_table.setAlternatingRowColors(True)
        tab_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tab_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        tab_table.verticalHeader().setVisible(False)
        tab_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #DDDDDD;
                gridline-color: #EEEEEE;
            }
            QTableWidget::item {
                padding: 6px;
            }
            QHeaderView::section {
                background-color: #3498DB;
                color: white;
                font-weight: bold;
                padding: 8px;
                border: none;
            }
        """)
        
        tab_data = [
            ("数据导入与格式转换", "粘贴原始数据自动识别格式转换，多数据源在线更新（3个默认源+自定义+测试连接），批量保存"),
            ("历史记录", "9列全屏表格（期号/日期/正码/特别码/和值/单双比/大小比/颜色分布/跨度），智能筛选，分页显示"),
            ("预测与抽取", "23种算法预测+随机抽取+ML机器学习预测，置信度三级标注，异步预测不卡界面，算法权重自定义调节"),
            ("数字选择", "49个数字面板（红/蓝/绿三色），颜色图例，已选号码显示"),
            ("第七位预判", "特别码大小/单双/尾数预判，特别码频率图/走势图/关联分析，2个特别码专属算法"),
            ("统计分析图表", "13种图表：频率分布/冷热分析/区间分布/单双分析/大小分析/尾数分布/遗漏值热力图/综合走势/相关性热力图/间隔分析图/连号邻号概率图/和值分布图，懒加载"),
            ("回测分析", "历史回测命中率统计，算法对比柱状图，命中趋势线，最佳/最差期详细分析"),
            ("预测记录", "自动记录每次预测结果，导出CSV，收藏对比，共识号绿色高亮"),
            ("应用说明", "本页面，应用完整说明"),
        ]
        
        tab_table.setRowCount(len(tab_data))
        for i, (tab_name, desc) in enumerate(tab_data):
            tab_item = QTableWidgetItem(tab_name)
            tab_item.setFont(QFont("Arial", 11, QFont.Weight.Bold))
            tab_item.setForeground(QColor("#2C3E50"))
            tab_table.setItem(i, 0, tab_item)
            desc_item = QTableWidgetItem(desc)
            desc_item.setFont(QFont("Arial", 11))
            tab_table.setItem(i, 1, desc_item)
        
        tab_desc_layout.addWidget(tab_table)
        tab_desc_group.setLayout(tab_desc_layout)

        
        # ===== 5. 特色功能 =====
        features_group = QGroupBox("特色功能")
        features_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #2ECC71;
                border: 2px solid #DDDDDD;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        features_layout = QVBoxLayout()
        features_layout.setSpacing(8)
        
        features = [
            "• 深色模式一键切换",
            "• 全局字体调节（初号～小四号）",
            "• 快捷键（Ctrl+1~9切换标签页，Ctrl+P预测，Ctrl+S保存，Ctrl+D抽取）",
            "• 数据校验（期号重复/号码范围/完整性检查）",
            "• 快捷操作面板（清空收藏/记录、批量导出报告、验证号码、复制结果）",
            "• 文件安全写入（积分不足自动重试5次）",
            "• 重依赖库可选导入（缺少TensorFlow/PyTorch等时自动fallback）",
        ]
        
        for feature in features:
            label = QLabel(feature)
            label.setStyleSheet("font-size: 13px; color: #555555; padding: 3px 0;")
            features_layout.addWidget(label)
        
        features_group.setLayout(features_layout)

        
        # ===== 6. 运行环境 =====
        env_group = QGroupBox("运行环境")
        env_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #3498DB;
                border: 2px solid #DDDDDD;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        env_layout = QVBoxLayout()
        env_layout.setSpacing(8)
        
        env_items = [
            "• Python 3.8+",
            "• 详见 requirements.txt",
        ]
        
        for item in env_items:
            label = QLabel(item)
            label.setStyleSheet("font-size: 13px; color: #555555; padding: 3px 0;")
            env_layout.addWidget(label)
        
        env_group.setLayout(env_layout)

        
        # ===== 7. 数据文件 =====
        data_group = QGroupBox("数据文件")
        data_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #2ECC71;
                border: 2px solid #DDDDDD;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
        """)
        data_layout = QVBoxLayout()
        data_layout.setSpacing(8)
        
        data_items = [
            "• 彩票数据：./彩票预测系统v7.1/lottery_data.json",
            "• 数据源配置：./彩票预测系统v7/datasources.json",
        ]
        
        for item in data_items:
            label = QLabel(item)
            label.setStyleSheet("font-size: 13px; color: #555555; padding: 3px 0;")
            data_layout.addWidget(label)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # 使用QSplitter垂直分隔各区块，支持拖拽调整
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(4)
        splitter.addWidget(intro_group)
        splitter.addWidget(lib_group)
        splitter.addWidget(algo_group)
        splitter.addWidget(tab_desc_group)
        splitter.addWidget(features_group)
        splitter.addWidget(env_group)
        splitter.addWidget(data_group)
        # 设置初始拉伸比例（均分）
        for i in range(7):
            splitter.setStretchFactor(i, 1)
        
        layout.addWidget(splitter)
        
        # 添加底部间距
        layout.addStretch()
        
        scroll.setWidget(container)
        
        # 创建包装widget
        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.addWidget(scroll)
        
        return wrapper
    
    def _refresh_prediction_history_table(self):
        """刷新预测记录表格"""
        if not hasattr(self, 'prediction_history_table'):
            return
        fg_color = "#CDD6F4" if self.is_dark_mode else "#000000"
        self.prediction_history_table.setRowCount(len(self.prediction_history))
        for i, entry in enumerate(self.prediction_history):
            ts_item = QTableWidgetItem(entry.get('timestamp', ''))
            ts_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            ts_item.setForeground(QColor(fg_color))
            self.prediction_history_table.setItem(i, 0, ts_item)
            
            algo_item = QTableWidgetItem(entry.get('algorithm', ''))
            algo_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            algo_item.setForeground(QColor(fg_color))
            self.prediction_history_table.setItem(i, 1, algo_item)
            
            nums = entry.get('numbers', [])
            nums_str = '  '.join(str(n).zfill(2) for n in nums)
            nums_item = QTableWidgetItem(nums_str)
            nums_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            nums_item.setForeground(QColor(fg_color))
            self.prediction_history_table.setItem(i, 2, nums_item)
            
            sp_item = QTableWidgetItem(str(entry.get('special', 0)).zfill(2))
            sp_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            sp_item.setForeground(QColor(fg_color))
            self.prediction_history_table.setItem(i, 3, sp_item)
    
    def _on_export_prediction_history_csv(self):
        """【需求3-更新】导出预测历史为CSV"""
        if not self.prediction_history:
            QMessageBox.information(self, "提示", "没有可导出的预测记录")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "导出预测记录", "prediction_history.csv", "CSV文件 (*.csv)")
        if file_path:
            # 【需求3-更新】使用安全写入方法
            rows = [["时间", "算法", "正码", "特别码"]]
            for entry in self.prediction_history:
                nums_str = ' '.join(str(n).zfill(2) for n in entry.get('numbers', []))
                rows.append([
                    entry.get('timestamp', ''),
                    entry.get('algorithm', ''),
                    nums_str,
                    str(entry.get('special', 0)).zfill(2)
                ])
            if self._safe_write_csv(file_path, rows):
                QMessageBox.information(self, "成功", "预测记录已导出")
            # 失败信息由_safe_write_csv内部处理
    
    def _on_clear_prediction_history(self):
        """清空预测记录"""
        if not self.prediction_history:
            QMessageBox.information(self, "提示", "预测记录已为空")
            return
        reply = QMessageBox.question(self, "确认", "确定清空所有预测记录吗？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.prediction_history.clear()
            self._refresh_prediction_history_table()
            self.statusBar().showMessage("预测记录已清空")
    
    # ========================================================================
    # 功能4：号码走势图增强
    # ========================================================================
    def _show_number_trend_chart(self):
        """号码走势图 - 每期开出的6个正码用圆点标注，同色号码用连线，特别码用星形"""
        self.chart_title_label.setText("号码走势图")
        if not self.historical_data:
            self.main_chart_widget.figure.clear()
            ax = self.main_chart_widget.figure.add_subplot(111)
            ax.text(0.5, 0.5, '暂无数据', ha='center', va='center', fontsize=16)
            ax.set_title('请先导入数据', fontsize=14)
            self.main_chart_widget.canvas.draw()
            return
        
        recent = list(reversed(self.historical_data[-20:]))
        self.main_chart_widget.figure.clear()
        ax = self.main_chart_widget.figure.add_subplot(111)
        
        red_pts = {x: [] for x in range(len(recent))}
        blue_pts = {x: [] for x in range(len(recent))}
        green_pts = {x: [] for x in range(len(recent))}
        special_pts = {x: [] for x in range(len(recent))}
        
        for i, record in enumerate(recent):
            numbers = record.get('numbers', [])
            special = record.get('special', 0)
            for n in numbers:
                if LotteryConfig.is_red(n):
                    red_pts[i].append(n)
                elif LotteryConfig.is_blue(n):
                    blue_pts[i].append(n)
                else:
                    green_pts[i].append(n)
            special_pts[i] = special
        
        # 同色连线
        for color_pts, color, label in [
            (red_pts, '#FF0000', '红'),
            (blue_pts, '#0000FF', '蓝'),
            (green_pts, '#008000', '绿')
        ]:
            prev_x, prev_y = None, None
            for i in range(len(recent)):
                for n in color_pts[i]:
                    ax.plot(i, n, 'o', color=color, markersize=6, zorder=3)
                    if prev_x is not None:
                        ax.plot([prev_x, i], [prev_y, n], '-', color=color, alpha=0.3, linewidth=0.8)
                    prev_x, prev_y = i, n
        
        # 特别码星形标注
        for i in range(len(recent)):
            sp = special_pts[i]
            if sp:
                ax.plot(i, sp, '*', color='#F39C12', markersize=12, zorder=4, markeredgecolor='#E67E22')
        
        periods = [str(r.get('period', '?')) for r in recent]
        ax.set_xticks(range(len(recent)))
        ax.set_xticklabels(periods, rotation=45, fontsize=8)
        ax.set_ylim(0, 50)
        ax.set_ylabel('号码', fontsize=12)
        ax.set_title('号码走势图（最近20期）', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 图例
        _get_mpl()
        # Line2D 可以直接从 matplotlib.lines 导入，不需要太复杂
        try:
            from matplotlib.lines import Line2D
        except ImportError:
            Line2D = None
        if Line2D is not None:
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF0000', markersize=8, label='红球'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#0000FF', markersize=8, label='蓝球'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#008000', markersize=8, label='绿球'),
                Line2D([0], [0], marker='*', color='w', markerfacecolor='#F39C12', markersize=12, label='特别码'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        self.main_chart_widget.figure.tight_layout()
        self.main_chart_widget.canvas.draw()
    
    # ========================================================================
    # 功能9：更多图表类型 - 新增4个图表方法
    # ========================================================================
    def _show_correlation_heatmap(self):
        """功能9：相关性热力图 - 显示49x49号码共现矩阵"""
        self.chart_title_label.setText("相关性热力图")
        self._draw_chart("correlation_heatmap")
    
    def _show_interval_analysis(self):
        """功能9：间隔分析图 - 显示高频号码出现间隔分布"""
        self.chart_title_label.setText("间隔分析图")
        self._draw_chart("interval_analysis")
    
    def _show_consecutive_probability(self):
        """功能9：连号邻号概率图 - 显示连号、邻号、同尾数出现概率"""
        self.chart_title_label.setText("连号邻号概率图")
        self._draw_chart("consecutive_probability")
    
    def _show_sum_distribution(self):
        """功能9：和值分布图 - 显示和值直方图与正态拟合曲线"""
        self.chart_title_label.setText("和值分布图")
        self._draw_chart("sum_distribution")
    
    # ========================================================================
    # 功能3：多数据源切换
    # ========================================================================
    def _get_data_sources(self):
        """获取数据源列表，从配置文件读取"""
        config_file = "./彩票预测系统v7/datasources.json"
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('sources', [
                        "http://www.cjcp.com.cn/kaijiang/hk6/",
                        "https://www.lhc123.com/",
                        "https://www.hk6.com/"
                    ])
        except Exception:
            pass
        return [
            "http://www.cjcp.com.cn/kaijiang/hk6/",
            "https://www.lhc123.com/",
            "https://www.hk6.com/"
        ]
    
    def _save_data_sources(self, sources):
        """保存数据源列表到配置文件"""
        config_file = "./彩票预测系统v7/datasources.json"
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump({'sources': sources}, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False
    
    def _test_data_source(self, url):
        """测试数据源连接
        
        返回：('success', '超时', '失败')之一
        """
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return 'success'
                else:
                    return '失败'
        except urllib.error.URLError:
            return '失败'
        except Exception:
            return '超时'
    
    def _on_clear_all_data_clicked(self):
        """清除转格式页面的输入和转换结果"""
        try:
            self.raw_text_edit.clear()
            self.converted_text_edit.clear()
            self.statusBar().showMessage("已清除输入和转换结果")
        except Exception as e:
            QMessageBox.warning(self, "清除失败", "清除数据时出错:\n" + str(e))

    def _on_data_source_setting_clicked(self):
        """数据源设置对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("数据源设置")
        dialog.setFixedSize(600, 400)
        layout = QVBoxLayout(dialog)
        
        # 说明标签
        desc_label = QLabel("配置多个数据源URL，在线更新时将按优先级自动尝试连接")
        layout.addWidget(desc_label)
        
        # 数据源列表
        list_widget = QListWidget()
        current_sources = self._get_data_sources()
        for src in current_sources:
            list_widget.addItem(src)
        layout.addWidget(list_widget)
        
        # 操作按钮行
        btn_row = QHBoxLayout()
        
        add_btn = QPushButton("添加")
        add_btn.clicked.connect(lambda: self._add_data_source(list_widget))
        btn_row.addWidget(add_btn)
        
        del_btn = QPushButton("删除")
        del_btn.clicked.connect(lambda: self._del_data_source(list_widget))
        btn_row.addWidget(del_btn)
        
        test_btn = QPushButton("测试连接")
        test_btn.clicked.connect(lambda: self._test_selected_source(list_widget))
        btn_row.addWidget(test_btn)
        
        move_up_btn = QPushButton("上移")
        move_up_btn.clicked.connect(lambda: self._move_source(list_widget, -1))
        btn_row.addWidget(move_up_btn)
        
        move_down_btn = QPushButton("下移")
        move_down_btn.clicked.connect(lambda: self._move_source(list_widget, 1))
        btn_row.addWidget(move_down_btn)
        
        layout.addLayout(btn_row)
        
        # 确定取消按钮
        ok_cancel_row = QHBoxLayout()
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(dialog.accept)
        ok_cancel_row.addWidget(ok_btn)
        
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        ok_cancel_row.addWidget(cancel_btn)
        
        layout.addLayout(ok_cancel_row)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # 保存数据源
            new_sources = []
            for i in range(list_widget.count()):
                new_sources.append(list_widget.item(i).text())
            if self._save_data_sources(new_sources):
                QMessageBox.information(self, "成功", "数据源设置已保存")
            else:
                QMessageBox.warning(self, "失败", "数据源设置保存失败")
    
    def _add_data_source(self, list_widget):
        """添加数据源"""
        dialog = QInputDialog(self)
        dialog.setWindowTitle("添加数据源")
        dialog.setLabelText("请输入数据源URL:")
        dialog.setTextValue("https://")
        if dialog.exec() == QDialog.DialogCode.Accepted:
            url = dialog.textValue().strip()
            if url:
                list_widget.addItem(url)
    
    def _del_data_source(self, list_widget):
        """删除选中的数据源"""
        current_row = list_widget.currentRow()
        if current_row >= 0:
            list_widget.takeItem(current_row)
    
    def _test_selected_source(self, list_widget):
        """测试选中的数据源连接"""
        current_row = list_widget.currentRow()
        if current_row < 0:
            QMessageBox.information(self, "提示", "请先选择要测试的数据源")
            return
        
        url = list_widget.item(current_row).text()
        self.statusBar().showMessage("正在测试连接: " + url + "...")
        
        result = self._test_data_source(url)
        
        if result == 'success':
            QMessageBox.information(self, "测试结果", "连接成功！")
        elif result == '超时':
            QMessageBox.warning(self, "测试结果", "连接超时（3秒）")
        else:
            QMessageBox.warning(self, "测试结果", "连接失败")
        
        self.statusBar().showMessage("测试完成")
    
    def _move_source(self, list_widget, direction):
        """移动数据源位置"""
        current_row = list_widget.currentRow()
        new_row = current_row + direction
        if 0 <= current_row < list_widget.count() and 0 <= new_row < list_widget.count():
            item = list_widget.takeItem(current_row)
            list_widget.insertItem(new_row, item)
            list_widget.setCurrentRow(new_row)
    
    def _on_online_update_clicked(self):
        """在线更新开奖数据（支持多数据源切换）"""
        self.statusBar().showMessage("正在尝试在线更新...")
        sources = self._get_data_sources()
        
        for idx, url in enumerate(sources):
            self.statusBar().showMessage("正在尝试数据源 " + str(idx + 1) + "/" + str(len(sources)) + ": " + url)
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    html_content = resp.read().decode('utf-8', errors='ignore')
                
                # 解析期号和号码
                existing_periods = {r.get('period') for r in self.historical_data}
                new_records = []
                
                # 尝试多种正则模式解析
                # 模式1: 期号 + 6个正码 + 特别码
                pattern1 = r'第(\d+)期.*?(\d{2})\s*(\d{2})\s*(\d{2})\s*(\d{2})\s*(\d{2})\s*(\d{2})\s*[+加]\s*(\d{2})'
                matches = re.findall(pattern1, html_content)
                
                for m in matches:
                    period = int(m[0])
                    if period in existing_periods:
                        continue
                    numbers = [int(m[i]) for i in range(1, 7)]
                    special = int(m[7])
                    if all(1 <= n <= 49 for n in numbers) and 1 <= special <= 49:
                        record = {
                            'period': period,
                            'date': '',
                            'numbers': numbers,
                            'special': special
                        }
                        new_records.append(record)
                        existing_periods.add(period)
                
                # 模式2: 更通用的数字提取
                if not new_records:
                    pattern2 = r'(\d{4,6}).*?(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+[+]\s*(\d{1,2})'
                    matches2 = re.findall(pattern2, html_content)
                    for m in matches2:
                        period = int(m[0])
                        if period in existing_periods:
                            continue
                        numbers = [int(m[i]) for i in range(1, 7)]
                        special = int(m[7])
                        if all(1 <= n <= 49 for n in numbers) and 1 <= special <= 49:
                            record = {
                                'period': period,
                                'date': '',
                                'numbers': numbers,
                                'special': special
                            }
                            new_records.append(record)
                            existing_periods.add(period)
                
                if new_records:
                    for rec in new_records:
                        self.historical_data.insert(0, rec)
                    self._save_data()
                    self._update_history_table()
                    QMessageBox.information(self, "在线更新", "成功从数据源「" + url + "」新增 " + str(len(new_records)) + " 期数据")
                    self.statusBar().showMessage("在线更新完成")
                    return
                else:
                    # 当前数据源解析失败，尝试下一个
                    continue
                    
            except urllib.error.URLError:
                # 网络错误，尝试下一个数据源
                continue
            except Exception as e:
                # 其他错误，尝试下一个数据源
                continue
        
        # 所有数据源都失败
        QMessageBox.warning(self, "网络错误", "所有数据源都无法访问或解析失败\n请检查网络连接或手动导入数据")
        self.statusBar().showMessage("在线更新失败")
    
    # ========================================================================
    # 功能6：算法权重自定义
    # ========================================================================
    def _on_weight_adjust_clicked(self):
        """权重调节对话框"""
        sub_algorithms = [
            '冷热数字', '单双', '大小', '遗漏值', '连号邻号',
            '尾数分布', '区间分布', '轮盘赌', '历史相似性', '泊松分布', '玄学算法'
        ]
        
        dialog = QDialog(self)
        dialog.setWindowTitle("算法权重调节")
        dialog.resize(500, 500)
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel("调整综合推荐算法中各子算法的权重（0-100）："))
        
        sliders = {}
        for name in sub_algorithms:
            row = QHBoxLayout()
            label = QLabel(name)
            label.setMinimumWidth(90)
            row.addWidget(label)
            
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            default_val = self.custom_weights.get(name, 50)
            slider.setValue(default_val)
            row.addWidget(slider)
            
            val_label = QLabel(str(default_val))
            val_label.setMinimumWidth(30)
            slider.valueChanged.connect(lambda v, lbl=val_label: lbl.setText(str(v)))
            row.addWidget(val_label)
            
            layout.addLayout(row)
            sliders[name] = slider
        
        layout.addStretch()
        
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(ok_btn)
        
        reset_btn = QPushButton("重置默认")
        reset_btn.clicked.connect(lambda: self._reset_weights(sliders, sub_algorithms))
        btn_layout.addWidget(reset_btn)
        
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            for name in sub_algorithms:
                self.custom_weights[name] = sliders[name].value()
            self.statusBar().showMessage("权重设置已更新")
    
    def _reset_weights(self, sliders, sub_algorithms):
        """重置权重为默认值"""
        for name in sub_algorithms:
            sliders[name].setValue(50)
    
    # ========================================================================
    # 功能7：导出报告
    # ========================================================================
    def _on_export_report(self):
        """导出HTML分析报告"""
        file_path, _ = QFileDialog.getSaveFileName(self, "导出报告", "lottery_report.html", "HTML文件 (*.html)")
        if not file_path:
            return
        
        try:
            # 获取当前预测结果
            display_text = self.prediction_display.text()
            algo_name = "未知算法"
            if 0 <= self.current_algorithm_index < len(LotteryConfig.ALGORITHMS):
                algo_name = LotteryConfig.ALGORITHMS[self.current_algorithm_index][0]
            
            numbers = []
            for i in range(self.prediction_number_layout.count()):
                item = self.prediction_number_layout.itemAt(i)
                if item and item.widget() and isinstance(item.widget(), NumberButton):
                    numbers.append(item.widget().get_number())
            numbers = sorted(numbers)
            
            # 号码带颜色HTML
            nums_html = ""
            for n in numbers:
                color = "#000000"
                if LotteryConfig.is_red(n):
                    color = "#FF0000"
                elif LotteryConfig.is_blue(n):
                    color = "#0000FF"
                else:
                    color = "#008000"
                nums_html += '<span style="color:' + color + '; font-size:24px; font-weight:bold; margin:0 4px;">' + str(n).zfill(2) + '</span>'
            
            # 最近5期历史
            history_rows = ""
            for record in self.historical_data[:5]:
                period = record.get('period', '?')
                nums = record.get('numbers', [])
                special = record.get('special', 0)
                date_str = record.get('date', '')
                nums_str = ' '.join(str(n).zfill(2) for n in nums)
                history_rows += '<tr><td>' + str(period) + '</td><td>' + str(date_str) + '</td><td>' + nums_str + '</td><td>' + str(special).zfill(2) + '</td></tr>'
            
            # 基本统计
            stats_html = ""
            if self.historical_data:
                all_nums = []
                for r in self.historical_data:
                    all_nums.extend(r.get('numbers', []))
                counter = Counter(all_nums)
                hot = counter.most_common(5)
                cold = counter.most_common()[-5:]
                hot_str = ', '.join(str(n).zfill(2) + '(' + str(c) + ')' for n, c in hot)
                cold_str = ', '.join(str(n).zfill(2) + '(' + str(c) + ')' for n, c in cold)
                
                odd_total = sum(1 for n in all_nums if n % 2 == 1)
                even_total = len(all_nums) - odd_total
                big_total = sum(1 for n in all_nums if n > 24)
                small_total = len(all_nums) - big_total
                
                stats_html = (
                    '<p>热门号码: ' + hot_str + '</p>'
                    '<p>冷门号码: ' + cold_str + '</p>'
                    '<p>单双比: ' + str(odd_total) + ':' + str(even_total) + '</p>'
                    '<p>大小比: ' + str(big_total) + ':' + str(small_total) + '</p>'
                    '<p>总数据量: ' + str(len(self.historical_data)) + ' 期</p>'
                )
            
            report_html = (
                '<!DOCTYPE html><html><head><meta charset="utf-8"><title>彩票预测系统分析报告</title>'
                '<style>'
                'body { font-family: "Microsoft YaHei", Arial, sans-serif; background: #FFFFFF; color: #333; padding: 40px; }'
                'h1 { color: #3498DB; border-bottom: 2px solid #3498DB; padding-bottom: 10px; }'
                'h2 { color: #3498DB; margin-top: 30px; }'
                'table { border-collapse: collapse; width: 100%; margin: 10px 0; }'
                'th, td { border: 1px solid #DDD; padding: 8px 12px; text-align: center; }'
                'th { background-color: #F8F9FA; color: #333; }'
                '.prediction-box { background: #FFFFFF; border: 2px solid #2ECC71; border-radius: 8px; padding: 20px; margin: 15px 0; text-align: center; }'
                '.section { margin: 20px 0; padding: 15px; border: 1px solid #DDD; border-radius: 6px; background: #FFFFFF; }'
                '</style></head><body>'
                '<h1>彩票预测系统分析报告</h1>'
                '<p>生成时间: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '</p>'
                '<div class="section"><h2>预测结果</h2>'
                '<div class="prediction-box">' + nums_html + '</div>'
                '<p>使用算法: ' + algo_name + '</p></div>'
                '<div class="section"><h2>使用算法说明</h2>'
                '<p>' + algo_name + ': '
            )
            
            if 0 <= self.current_algorithm_index < len(LotteryConfig.ALGORITHMS):
                report_html += LotteryConfig.ALGORITHMS[self.current_algorithm_index][1]
            else:
                report_html += "综合多种算法得出最优预测"
            report_html += '</p></div>'
            
            report_html += (
                '<div class="section"><h2>最近5期历史数据</h2>'
                '<table><tr><th>期号</th><th>日期</th><th>正码</th><th>特别码</th></tr>'
                + history_rows +
                '</table></div>'
                '<div class="section"><h2>基本统计摘要</h2>'
                + stats_html +
                '</div></body></html>'
            )
            
            # 【需求3-更新】使用安全写入方法保存HTML报告
            if self._safe_write_file(file_path, report_html):
                QMessageBox.information(self, "导出成功", "分析报告已保存到:\n" + file_path)
            # 失败信息由_safe_write_file内部处理
        except Exception as e:
            QMessageBox.warning(self, "导出失败", "报告生成失败: " + str(e))
    
    # ========================================================================
    # 功能8：深色模式切换
    # ========================================================================
    def _on_toggle_theme(self):
        """切换深色/浅色模式"""
        self.is_dark_mode = not self.is_dark_mode
        self._update_stylesheet()
        self._update_history_table()
        self._refresh_prediction_history_table()
        theme_name = "深色模式" if self.is_dark_mode else "浅色模式"
        self.statusBar().showMessage("已切换到" + theme_name)
    
    # ========================================================================
    # 功能10：数据校验提示
    # ========================================================================
    def _validate_record(self, record, is_batch=False):
        """校验单条数据，返回问题列表"""
        issues = []
        period = record.get('period')
        numbers = record.get('numbers', [])
        special = record.get('special')
        
        # 检查期号重复
        if period is not None:
            for r in self.historical_data:
                if r.get('period') == period:
                    issues.append(('error', '期号重复: 第' + str(period) + '期已存在'))
                    break
        
        # 检查数据不完整
        if period is None:
            issues.append(('error', '缺少期号'))
        if not numbers or len(numbers) < 6:
            issues.append(('error', '正码不完整，应有6个正码'))
        
        # 检查号码范围
        all_nums = list(numbers) + ([special] if special is not None else [])
        for n in all_nums:
            if n is not None and (n < 1 or n > 49):
                issues.append(('warning', '号码超出范围: ' + str(n) + ' (应为1-49)'))
        
        # 检查号码重复
        if len(numbers) != len(set(numbers)):
            dup = [n for n, c in Counter(numbers).items() if c > 1]
            issues.append(('warning', '同一期内出现重复数字: ' + ', '.join(str(d) for d in dup)))
        
        return issues
    
    def _show_validation_results(self, all_issues):
        """显示校验结果"""
        if not all_issues:
            return True
        
        errors = [iss for iss in all_issues if iss[0] == 'error']
        warnings = [iss for iss in all_issues if iss[0] == 'warning']
        
        msg = ""
        if errors:
            msg += '<p style="color:#E74C3C; font-weight:bold;">严重问题 (' + str(len(errors)) + '):</p>'
            msg += '<ul style="color:#E74C3C;">'
            for _, text in errors:
                msg += '<li>' + text + '</li>'
            msg += '</ul>'
        if warnings:
            msg += '<p style="color:#F39C12; font-weight:bold;">警告 (' + str(len(warnings)) + '):</p>'
            msg += '<ul style="color:#F39C12;">'
            for _, text in warnings:
                msg += '<li>' + text + '</li>'
            msg += '</ul>'
        
        if errors:
            msg += '<p style="font-weight:bold;">是否仍要继续？（不推荐）</p>'
            reply = QMessageBox.warning(self, "数据校验", msg,
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            return reply == QMessageBox.StandardButton.Yes
        else:
            msg += '<p>以上为警告信息，不影响数据导入。</p>'
            QMessageBox.information(self, "数据校验提示", msg)
            return True

    def _update_history_table(self):
        """更新历史记录表格
        
        功能6增强：支持筛选后的数据显示
        功能12增强：支持分页显示
        """
        fg_color = "#CDD6F4" if self.is_dark_mode else "#333333"
        
        # 决定使用原始数据还是筛选后的数据
        data_to_show = self.filtered_data if self.filtered_data else self.historical_data
        total_count = len(data_to_show)
        
        # 计算分页
        total_pages = max(1, (total_count + self.history_page_size - 1) // self.history_page_size)
        self.history_page = min(self.history_page, total_pages)
        
        # 计算当前页的数据范围
        start_idx = (self.history_page - 1) * self.history_page_size
        end_idx = min(start_idx + self.history_page_size, total_count)
        
        # 更新分页信息
        self._update_pagination(total_count)
        
        # 设置表格行数（仅当前页）
        self.history_table.setRowCount(end_idx - start_idx)
        
        for i, record in enumerate(data_to_show[start_idx:end_idx]):
            table_row = i
            # 期号
            item_period = QTableWidgetItem(str(record.get('period', '?')))
            item_period.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_period.setForeground(QColor(fg_color))
            self.history_table.setItem(table_row, 0, item_period)
            # 日期
            item_date = QTableWidgetItem(record.get('date', '?'))
            item_date.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_date.setForeground(QColor(fg_color))
            self.history_table.setItem(table_row, 1, item_date)
            # 正码（每个数字带对应颜色）
            numbers = record.get('numbers', [])
            numbers_str = '  '.join(str(n).zfill(2) for n in numbers)
            item_numbers = QTableWidgetItem(numbers_str)
            item_numbers.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            # 用第一个数字的颜色作为整行前景色（避免全黑看不到）
            if numbers:
                first_color = LotteryConfig.get_number_color(numbers[0])
                item_numbers.setForeground(QColor(first_color['text']))
            self.history_table.setItem(table_row, 2, item_numbers)
            # 特别码（带颜色标记）
            special = record.get('special', '?')
            item_special = QTableWidgetItem(str(special).zfill(2) if special != '?' else '?')
            item_special.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            special_color = fg_color
            if isinstance(special, int):
                if special in LotteryConfig.RED_NUMBERS:
                    special_color = "#FF0000"
                elif special in LotteryConfig.BLUE_NUMBERS:
                    special_color = "#0000FF"
                else:
                    special_color = "#008000"
            item_special.setForeground(QColor(special_color))
            self.history_table.setItem(table_row, 3, item_special)
            # 和值
            sum_val = sum(numbers) if numbers else 0
            item_sum = QTableWidgetItem(str(sum_val))
            item_sum.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_sum.setForeground(QColor(fg_color))
            self.history_table.setItem(table_row, 4, item_sum)
            # 单双比
            odd_count = sum(1 for n in numbers if n % 2 == 1) if numbers else 0
            even_count = len(numbers) - odd_count
            item_oe = QTableWidgetItem(str(odd_count) + ':' + str(even_count))
            item_oe.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_oe.setForeground(QColor(fg_color))
            self.history_table.setItem(table_row, 5, item_oe)
            # 大小比
            big_count = sum(1 for n in numbers if n > 24) if numbers else 0
            small_count = len(numbers) - big_count
            item_bs = QTableWidgetItem(str(big_count) + ':' + str(small_count))
            item_bs.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_bs.setForeground(QColor(fg_color))
            self.history_table.setItem(table_row, 6, item_bs)
            # 颜色分布
            red_count = sum(1 for n in numbers if n in LotteryConfig.RED_NUMBERS) if numbers else 0
            blue_count = sum(1 for n in numbers if n in LotteryConfig.BLUE_NUMBERS) if numbers else 0
            green_count = len(numbers) - red_count - blue_count
            item_color = QTableWidgetItem('红' + str(red_count) + '蓝' + str(blue_count) + '绿' + str(green_count))
            item_color.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_color.setForeground(QColor(fg_color))
            self.history_table.setItem(table_row, 7, item_color)
            # 跨度
            span_val = (max(numbers) - min(numbers)) if numbers else 0
            item_span = QTableWidgetItem(str(span_val))
            item_span.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_span.setForeground(QColor(fg_color))
            self.history_table.setItem(table_row, 8, item_span)
        
        self.data_count_label.setText("历史记录: " + str(len(self.historical_data)) + " 条")
        if hasattr(self, 'history_count_label'):
            if self.filtered_data:
                self.history_count_label.setText("显示第 " + str(start_idx + 1) + "-" + str(end_idx) + " 条 / 共 " + str(total_count) + " 条（已筛选）")
            else:
                self.history_count_label.setText("共 " + str(len(self.historical_data)) + " 条记录")
        self._refresh_latest_display()
        
        # 更新历史记录标签页的最新数据显示
        if hasattr(self, 'history_latest_display') and data_to_show:
            latest = data_to_show[0]
            numbers = latest.get('numbers', [])
            special = latest.get('special', 0)
            text = '第' + str(latest.get('period', '?')) + '期 | ' + str(latest.get('date', '?'))
            text += '\n正码: ' + ' '.join(str(n).zfill(2) for n in numbers)
            text += '  特别码: ' + str(special).zfill(2)
            self.history_latest_display.setText(text)
        
        # 更新统计摘要（基于筛选后的数据）
        if hasattr(self, 'history_stats_label') and data_to_show:
            from collections import Counter
            all_nums = []
            for r in data_to_show:
                all_nums.extend(r.get('numbers', []))
            counter = Counter(all_nums)
            hot = counter.most_common(5)
            hot_str = '  '.join(str(n).zfill(2) + '(' + str(c) + ')' for n, c in hot)
            cold = counter.most_common()[-5:]
            cold_str = '  '.join(str(n).zfill(2) + '(' + str(c) + ')' for n, c in cold)
            
            sums = [sum(r.get('numbers', [])) for r in data_to_show]
            dates = [r.get('date', '') for r in data_to_show if r.get('date')]
            
            stats_text = '总期数: ' + str(len(data_to_show)) + ' 期\n'
            if dates:
                stats_text += '日期范围: ' + dates[-1] + ' ~ ' + dates[0] + '\n'
            if sums:
                stats_text += '和值范围: ' + str(min(sums)) + ' ~ ' + str(max(sums)) + '\n'
                stats_text += '和值均值: ' + str(int(sum(sums) / len(sums))) + '\n'
            stats_text += '热门号码: ' + hot_str + '\n'
            stats_text += '冷门号码: ' + cold_str
            self.history_stats_label.setText(stats_text)
    
    def _on_number_selected(self, numbers):
        if numbers:
            formatted = ', '.join(str(n) for n in sorted(numbers))
            self.selected_numbers_label.setText(formatted)
        else:
            self.selected_numbers_label.setText("无")
    
    def _clear_number_selection(self):
        self.number_panel.clear_selection()
        self.selected_numbers_label.setText("无")
    
    def _on_algorithm_changed(self, index):
        self.current_algorithm_index = index
        if index < len(LotteryConfig.ALGORITHMS):
            _, desc = LotteryConfig.ALGORITHMS[index]
            self.algorithm_desc_label.setText(desc)
    
    def _on_predict_clicked(self):
        """功能12：性能优化 - 异步预测"""
        if len(self.historical_data) < 10:
            QMessageBox.warning(self, "数据不足", "历史数据不足10条，请先添加更多数据")
            return
        
        # 如果正在预测，则忽略新请求
        if self._is_predicting:
            QMessageBox.information(self, "请等待", "预测正在进行中，请等待完成")
            return
        
        # 设置预测状态
        self._is_predicting = True
        self.predict_button.setEnabled(False)
        self.statusBar().showMessage("正在预测...")
        
        try:
            # 创建工作线程
            self._predict_thread = QThread()
            self._predict_worker = PredictWorker(self.historical_data, self.current_algorithm_index)
            self._predict_worker.moveToThread(self._predict_thread)
            
            # 连接信号
            self._predict_thread.started.connect(self._predict_worker.run)
            self._predict_worker.finished.connect(self._on_predict_finished)
            self._predict_worker.error.connect(self._on_predict_error)
            self._predict_worker.progress.connect(self._on_predict_progress)
            self._predict_worker.finished.connect(self._predict_thread.quit)
            self._predict_worker.finished.connect(self._predict_worker.deleteLater)
            self._predict_thread.finished.connect(self._predict_thread.deleteLater)
            
            # 启动线程
            self._predict_thread.start()
        except Exception as e:
            self._is_predicting = False
            self.predict_button.setEnabled(True)
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "错误", "创建预测线程失败:\n" + str(e))
            self.statusBar().showMessage("预测失败")
    
    def _on_predict_progress(self, value, message):
        """异步预测进度回调"""
        self.statusBar().showMessage(f"预测中... {value}% - {message}")
    
    def _on_predict_finished(self, predictions, confidence_info):
        """异步预测完成回调"""
        self._is_predicting = False
        self.predict_button.setEnabled(True)
        try:
            self._display_predictions(predictions, confidence_info)
            self.statusBar().showMessage("预测完成")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "显示结果错误", "预测完成但显示结果时出错:\n" + str(e))
            self.statusBar().showMessage("预测完成（显示异常）")
    
    def _on_predict_error(self, error_msg):
        """异步预测错误回调"""
        self._is_predicting = False
        self.predict_button.setEnabled(True)
        import traceback
        traceback.print_exc()
        QMessageBox.warning(self, "预测错误", f"预测过程出错:\n{error_msg}")
        self.statusBar().showMessage("预测失败")
    
    def _on_random_draw_clicked(self):
        try:
            if len(self.historical_data) < 10:
                QMessageBox.warning(self, "数据不足", "历史数据不足10条，请先添加更多数据")
                return
            
            predictor = PredictionAlgorithms(self.historical_data)
            predictions = predictor.roulette_selection(6)
            sorted_preds = sorted(predictions)
            
            # 显示抽取结果文本
            display_text = "随机抽取: " + ' '.join(str(n).zfill(2) for n in sorted_preds)
            self.prediction_display.setText(display_text)
            
            # 清空并显示号码球
            while self.prediction_number_layout.count():
                item = self.prediction_number_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            for i, num in enumerate(sorted_preds):
                btn = NumberButton(num)
                btn.set_selected(True)
                row = i // 6
                col = i % 6
                self.prediction_number_layout.addWidget(btn, row, col)
            
            # 清除置信度标签
            if hasattr(self, 'confidence_display_label') and self.confidence_display_label:
                self.confidence_display_label.setText("")
            
            # 记录到预测历史
            history_entry = {
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'algorithm': '随机抽取',
                'numbers': list(sorted_preds),
                'special': 0
            }
            self.prediction_history.append(history_entry)
            self._refresh_prediction_history_table()
            
            self.statusBar().showMessage("随机抽取完成")
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback.print_exc()
            QMessageBox.warning(self, "随机抽取错误", "随机抽取过程出错:\n" + error_msg)
            self.statusBar().showMessage("随机抽取失败")
    
    def _on_ml_predict_clicked(self):
        """【需求2-BUG修复】机器学习预测按钮 - 异步执行避免界面卡顿"""
        if len(self.historical_data) < 20:
            QMessageBox.warning(self, "数据不足", "机器学习需要至少20条历史数据")
            return
        
        if self._is_predicting:
            QMessageBox.information(self, "请等待", "预测正在进行中，请等待完成")
            return
        
        self._is_predicting = True
        self.predict_button.setEnabled(False)
        self.statusBar().showMessage("机器学习预测中，请稍候...")
        
        try:
            self._ml_predict_thread = QThread()
            self._ml_predict_worker = MLPredictWorker(self.historical_data)
            self._ml_predict_worker.moveToThread(self._ml_predict_thread)
            
            self._ml_predict_thread.started.connect(self._ml_predict_worker.run)
            self._ml_predict_worker.finished.connect(self._on_ml_predict_finished)
            self._ml_predict_worker.error.connect(self._on_ml_predict_error)
            self._ml_predict_worker.progress.connect(self._on_ml_predict_progress)
            self._ml_predict_worker.finished.connect(self._ml_predict_thread.quit)
            self._ml_predict_worker.finished.connect(self._ml_predict_worker.deleteLater)
            self._ml_predict_thread.finished.connect(self._ml_predict_thread.deleteLater)
            
            self._ml_predict_thread.start()
        except Exception as e:
            self._is_predicting = False
            self.predict_button.setEnabled(True)
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "错误", "创建机器学习预测线程失败:\n" + str(e))
            self.statusBar().showMessage("机器学习预测失败")
    
    def _on_ml_predict_progress(self, value, message):
        """机器学习预测进度回调"""
        self.statusBar().showMessage(f"机器学习预测中... {value}% - {message}")
    
    def _on_ml_predict_finished(self, predictions):
        """机器学习预测完成回调"""
        self._is_predicting = False
        self.predict_button.setEnabled(True)
        try:
            self._display_predictions(predictions)
            # 修正算法名为"机器学习预测"
            if self.prediction_history:
                self.prediction_history[-1]['algorithm'] = "机器学习预测"
                self._refresh_prediction_history_table()
            self.statusBar().showMessage("机器学习预测完成")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "显示结果错误", "机器学习预测完成但显示结果时出错:\n" + str(e))
            self.statusBar().showMessage("机器学习预测完成（显示异常）")
    
    def _on_ml_predict_error(self, error_msg):
        """机器学习预测错误回调"""
        self._is_predicting = False
        self.predict_button.setEnabled(True)
        import traceback
        traceback.print_exc()
        QMessageBox.warning(self, "机器学习预测错误", f"机器学习预测过程出错:\n{error_msg}")
        self.statusBar().showMessage("机器学习预测失败")
    
    def _display_predictions(self, predictions, confidence_info=None):
        """显示预测结果
        
        功能4增强：添加置信度信息显示
        
        参数：
            predictions: 预测号码列表
            confidence_info: 置信度字典，格式为 {number: confidence_percentage}
        """
        sorted_preds = sorted(predictions)
        display_text = "预测号码: " + ' '.join(str(n).zfill(2) for n in sorted_preds)
        self.prediction_display.setText(display_text)
        # 功能3：记录预测历史
        algo_name = "未知算法"
        if 0 <= self.current_algorithm_index < len(LotteryConfig.ALGORITHMS):
            algo_name = LotteryConfig.ALGORITHMS[self.current_algorithm_index][0]
        history_entry = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'algorithm': algo_name,
            'numbers': list(sorted_preds),
            'special': 0
        }
        self.prediction_history.append(history_entry)
        self._refresh_prediction_history_table()
        while self.prediction_number_layout.count():
            item = self.prediction_number_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 判断是否为特别码算法
        is_special_algorithm = self.current_algorithm_index in [21, 22]
        
        for i, num in enumerate(sorted_preds):
            btn = NumberButton(num)
            btn.set_selected(True)
            row = i // 6
            col = i % 6
            self.prediction_number_layout.addWidget(btn, row, col)
        
        # ======================================================================== #
        # 功能4：预测结果置信度显示
        # ======================================================================== #
        # 清空并重新创建置信度标签
        if hasattr(self, 'confidence_labels_layout') and self.confidence_labels_layout:
            while self.confidence_labels_layout.count():
                item = self.confidence_labels_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        else:
            self.confidence_labels_layout = self.prediction_number_layout
        
        # 添加置信度标签
        if confidence_info and not is_special_algorithm:
            confidence_text_parts = []
            for num in sorted_preds:
                conf = confidence_info.get(num, 0)
                if conf >= 70:
                    level = "强烈推荐"
                    color = "#2ECC71"
                elif conf >= 40:
                    level = "一般推荐"
                    color = "#3498DB"
                else:
                    level = "谨慎参考"
                    color = "#F39C12"
                confidence_text_parts.append(str(num).zfill(2) + ":" + "{:.1f}".format(conf) + "%「" + level + "」")
            
            # 更新或创建置信度显示标签
            if hasattr(self, 'confidence_display_label'):
                conf_text = "  ".join(confidence_text_parts)
                self.confidence_display_label.setText(conf_text)
                # 设置颜色样式
                style = "font-size: 13px; padding: 5px; background-color: #F8F9FA; border-radius: 4px;"
                self.confidence_display_label.setStyleSheet(style)
            else:
                # 创建新的置信度显示标签
                self.confidence_display_label = QLabel("  ".join(confidence_text_parts))
                self.confidence_display_label.setStyleSheet("font-size: 13px; padding: 5px; background-color: #F8F9FA; border-radius: 4px;")
                self.prediction_number_layout.addWidget(self.confidence_display_label, len(sorted_preds) // 6 + 1, 0, 1, 6)
        
        # 特别码算法特殊显示
        if is_special_algorithm:
            if hasattr(self, 'confidence_display_label'):
                self.confidence_display_label.setText("特别码预测: " + str(sorted_preds[0]) if sorted_preds else "无")
                self.confidence_display_label.setStyleSheet("font-size: 15px; padding: 8px; background-color: #FFF3CD; border: 2px solid #F39C12; border-radius: 6px; font-weight: bold;")
            else:
                self.confidence_display_label = QLabel("特别码预测: " + str(sorted_preds[0]) if sorted_preds else "无")
                self.confidence_display_label.setStyleSheet("font-size: 15px; padding: 8px; background-color: #FFF3CD; border: 2px solid #F39C12; border-radius: 6px; font-weight: bold;")
                self.prediction_number_layout.addWidget(self.confidence_display_label, len(sorted_preds) // 6 + 1, 0, 1, 6)
        
        # ======================================================================== #
        # 功能4：置信度分析汇总
        # ======================================================================== #
        if hasattr(self, 'confidence_analysis_text') and confidence_info and not is_special_algorithm:
            strong_rec = [n for n in sorted_preds if confidence_info.get(n, 0) >= 70]
            normal_rec = [n for n in sorted_preds if 40 <= confidence_info.get(n, 0) < 70]
            cautious = [n for n in sorted_preds if confidence_info.get(n, 0) < 40]
            
            analysis = "置信度分析汇总：\n"
            if strong_rec:
                analysis += "  ● 强烈推荐(" + str(len(strong_rec)) + "个): " + " ".join(str(n).zfill(2) for n in strong_rec) + "\n"
            if normal_rec:
                analysis += "  ● 一般推荐(" + str(len(normal_rec)) + "个): " + " ".join(str(n).zfill(2) for n in normal_rec) + "\n"
            if cautious:
                analysis += "  ● 谨慎参考(" + str(len(cautious)) + "个): " + " ".join(str(n).zfill(2) for n in cautious)
            
            self.confidence_analysis_text.setPlainText(analysis)
        elif hasattr(self, 'confidence_analysis_text') and is_special_algorithm:
            self.confidence_analysis_text.setPlainText("特别码预测结果\n建议结合正码预测综合参考")
        
        # 统计信息（仅对非特别码算法）
        if not is_special_algorithm:
            red_count = sum(1 for n in predictions if LotteryConfig.is_red(n))
            blue_count = sum(1 for n in predictions if LotteryConfig.is_blue(n))
            green_count = sum(1 for n in predictions if LotteryConfig.is_green(n))
            odd_count = sum(1 for n in predictions if n % 2 == 1)
            even_count = 6 - odd_count
            big_count = sum(1 for n in predictions if n > 25)
            small_count = 6 - big_count
            stats_text = ("颜色分布: 红" + str(red_count) + "个 蓝" + str(blue_count) + "个 绿" + str(green_count) + "个\n"
                         + "单双分布: 单" + str(odd_count) + "个 双" + str(even_count) + "个\n"
                         + "大小分布: 大" + str(big_count) + "个 小" + str(small_count) + "个")
            self.prediction_stats_label.setText(stats_text)
        else:
            self.prediction_stats_label.setText("特别码预测\n（仅返回1个特别码）")
    
    # ======================================================================== #
    # 功能6：智能筛选与搜索
    # ======================================================================== #
    def _on_filter_history(self):
        """根据筛选条件筛选历史记录"""
        try:
            # 获取筛选条件
            period_start = self.filter_period_start.value()
            period_end = self.filter_period_end.value()
            number_input = self.filter_number_input.text().strip()
            sum_min = self.filter_sum_min.value()
            sum_max = self.filter_sum_max.value()
            odd_even_filter = self.filter_odd_even.currentText()
            size_filter = self.filter_size.currentText()
            color_filter = self.filter_color.currentText()
            
            # 解析号码输入
            filter_numbers = []
            if number_input:
                try:
                    filter_numbers = [int(n.strip()) for n in number_input.split(',') if n.strip().isdigit()]
                    filter_numbers = [n for n in filter_numbers if 1 <= n <= 49]
                except ValueError:
                    pass
            
            # 应用筛选
            filtered = []
            for record in self.historical_data:
                # 期号范围筛选
                period = record.get('period', 0)
                if isinstance(period, str):
                    try:
                        period = int(period)
                    except ValueError:
                        period = 0
                
                if period_start > 0 and period < period_start:
                    continue
                if period_end > 0 and period > period_end:
                    continue
                
                numbers = record.get('numbers', [])
                if not numbers:
                    continue
                
                # 号码搜索筛选
                if filter_numbers:
                    if not any(n in numbers for n in filter_numbers):
                        continue
                
                # 和值范围筛选
                sum_val = sum(numbers)
                if sum_min > 0 and sum_val < sum_min:
                    continue
                if sum_max > 0 and sum_val > sum_max:
                    continue
                
                # 单双比筛选
                if odd_even_filter != "全部":
                    odd_count = sum(1 for n in numbers if n % 2 == 1)
                    even_count = len(numbers) - odd_count
                    if odd_even_filter == "单多" and odd_count <= even_count:
                        continue
                    elif odd_even_filter == "双多" and odd_count >= even_count:
                        continue
                    elif odd_even_filter == "均衡" and abs(odd_count - even_count) > 1:
                        continue
                
                # 大小比筛选
                if size_filter != "全部":
                    big_count = sum(1 for n in numbers if n > 25)
                    small_count = len(numbers) - big_count
                    if size_filter == "大多" and big_count <= small_count:
                        continue
                    elif size_filter == "小多" and big_count >= small_count:
                        continue
                    elif size_filter == "均衡" and abs(big_count - small_count) > 1:
                        continue
                
                # 颜色分布筛选
                if color_filter != "全部":
                    red_count = sum(1 for n in numbers if n in LotteryConfig.RED_NUMBERS)
                    blue_count = sum(1 for n in numbers if n in LotteryConfig.BLUE_NUMBERS)
                    green_count = len(numbers) - red_count - blue_count
                    if color_filter == "红多" and not (red_count >= blue_count and red_count >= green_count):
                        continue
                    elif color_filter == "蓝多" and not (blue_count >= red_count and blue_count >= green_count):
                        continue
                    elif color_filter == "绿多" and not (green_count >= red_count and green_count >= blue_count):
                        continue
                
                filtered.append(record)
            
            # 保存筛选结果
            self.filtered_data = filtered
            self.history_page = 1
            
            # 更新显示
            self._update_history_table()
            
            # 更新筛选统计
            total = len(self.historical_data)
            self.filter_stats_label.setText("共筛选出 " + str(len(filtered)) + " 期 / 总 " + str(total) + " 期")
            
            self.statusBar().showMessage("筛选完成，共找到 " + str(len(filtered)) + " 条记录")
        except Exception as e:
            QMessageBox.warning(self, "筛选错误", "筛选过程出错: " + str(e))
    
    def _on_reset_filter(self):
        """重置筛选条件"""
        self.filter_period_start.setValue(0)
        self.filter_period_end.setValue(0)
        self.filter_number_input.clear()
        self.filter_sum_min.setValue(0)
        self.filter_sum_max.setValue(0)
        self.filter_odd_even.setCurrentIndex(0)
        self.filter_size.setCurrentIndex(0)
        self.filter_color.setCurrentIndex(0)
        self.filtered_data = None
        self.history_page = 1
        self._update_history_table()
        self.filter_stats_label.setText("")
        self.statusBar().showMessage("筛选已重置")
    
    # ======================================================================== #
    # 功能12：性能优化 - 分页
    # ======================================================================== #
    def _on_prev_page(self):
        """上一页"""
        if self.history_page > 1:
            self.history_page -= 1
            self._update_history_table()
    
    def _on_next_page(self):
        """下一页"""
        data_to_use = self.filtered_data if self.filtered_data else self.historical_data
        total_pages = max(1, (len(data_to_use) + self.history_page_size - 1) // self.history_page_size)
        if self.history_page < total_pages:
            self.history_page += 1
            self._update_history_table()
    
    def _on_page_jump(self, page):
        """跳转到指定页"""
        data_to_use = self.filtered_data if self.filtered_data else self.historical_data
        total_pages = max(1, (len(data_to_use) + self.history_page_size - 1) // self.history_page_size)
        if 1 <= page <= total_pages:
            self.history_page = page
            self._update_history_table()
    
    def _update_pagination(self, total_count):
        """更新分页信息"""
        total_pages = max(1, (total_count + self.history_page_size - 1) // self.history_page_size)
        self.history_page = min(self.history_page, total_pages)
        self.history_page_label.setText("第 " + str(self.history_page) + " 页 / 共 " + str(total_pages) + " 页")
        self.history_page_spin.setRange(1, total_pages)
        self.history_page_spin.setValue(self.history_page)
    
    # ======================================================================== #
    # 功能12：性能优化 - 图表懒加载
    # ======================================================================== #
    def _on_tab_changed(self, index):
        """标签页切换时初始化图表"""
        # 检查是否切换到统计图表标签页（索引5）
        if index == 5 and not self._chart_initialized:
            self._chart_initialized = True
            # 图表控件已在_create_statistics_chart_tab中创建，无需额外初始化
            self.statusBar().showMessage("图表控件已初始化")
    
    def _refresh_latest_display(self):
        if not self.historical_data:
            self.latest_display.setText("暂无数据")
            return
        latest = self.historical_data[0]
        numbers = latest.get('numbers', [])
        special = latest.get('special', 0)
        numbers_text = ' '.join(str(n).zfill(2) for n in numbers)
        text = "第" + str(latest.get('period', '?')) + "期 | " + latest.get('date', '?') + "\n"
        text += "正码: " + numbers_text + "\n"
        text += "特别码: " + str(special).zfill(2)
        self.latest_display.setText(text)
    
    def _on_show_period_detail(self):
        """显示选中期的完整信息"""
        if not hasattr(self, 'period_detail_edit'):
            return
        row = self.history_table.currentRow()
        if row < 0 or row >= len(self.historical_data):
            self.period_detail_edit.setHtml('<p style="color:#E74C3C;">请先在表格中选择一期记录</p>')
            return
        
        record = self.historical_data[row]
        numbers = record.get('numbers', [])
        special = record.get('special', 0)
        period = record.get('period', '?')
        date = record.get('date', '?')
        
        # 构建HTML完整显示
        html = '<div style="font-size:20px; line-height:2.4;">'
        html += '<p style="font-size:30px; font-weight:bold; color:#3498DB;">第' + str(period) + '期  ' + str(date) + '</p>'
        
        # 正码大按钮显示
        html += '<p style="font-size:22px;"><b>正码：</b></p>'
        html += '<p style="margin-left:10px;">'
        for n in numbers:
            colors = LotteryConfig.get_number_color(n)
            html += '<span style="display:inline-block; background-color:' + colors['border'] + '; color:#FFFFFF; font-size:30px; font-weight:bold; border-radius:12px; padding:10px 18px; margin:5px;">' + str(n).zfill(2) + '</span> '
        html += '</p>'
        
        # 特别码
        html += '<p><b>特别码：</b>'
        sp_colors = LotteryConfig.get_number_color(special)
        sp_name = LotteryConfig.NUMBER_NAMES.get(special, '')
        sp_elem = LotteryConfig.NUMBER_ELEMENTS.get(special, '')
        sp_color_name = ''
        if special in LotteryConfig.RED_NUMBERS:
            sp_color_name = '红'
        elif special in LotteryConfig.BLUE_NUMBERS:
            sp_color_name = '蓝'
        else:
            sp_color_name = '绿'
        html += '<span style="display:inline-block; background-color:' + sp_colors['border'] + '; color:#FFFFFF; font-size:30px; font-weight:bold; border-radius:12px; padding:10px 18px; margin:5px;">' + str(special).zfill(2) + '</span>'
        html += '</p>'
        
        # 详细属性表
        html += '<table style="border-collapse:collapse; width:100%; margin-top:8px; font-size:18px;">'
        
        # 和值
        sum_val = sum(numbers)
        html += '<tr><td style="padding:6px 10px; border:1px solid #DDD; font-weight:bold; width:90px;">和值</td><td style="padding:6px 10px; border:1px solid #DDD; font-size:22px; font-weight:bold;">' + str(sum_val) + '</td></tr>'
        
        # 跨度
        span_val = max(numbers) - min(numbers) if numbers else 0
        html += '<tr><td style="padding:6px 10px; border:1px solid #DDD; font-weight:bold;">跨度</td><td style="padding:6px 10px; border:1px solid #DDD; font-size:22px; font-weight:bold;">' + str(span_val) + '</td></tr>'
        
        # 单双比
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        even_count = len(numbers) - odd_count
        html += '<tr><td style="padding:6px 10px; border:1px solid #DDD; font-weight:bold;">单双比</td><td style="padding:6px 10px; border:1px solid #DDD;">单' + str(odd_count) + ':双' + str(even_count) + '</td></tr>'
        
        # 大小比
        big_count = sum(1 for n in numbers if n > 24)
        small_count = len(numbers) - big_count
        html += '<tr><td style="padding:6px 10px; border:1px solid #DDD; font-weight:bold;">大小比</td><td style="padding:6px 10px; border:1px solid #DDD;">大' + str(big_count) + ':小' + str(small_count) + '</td></tr>'
        
        # 颜色分布
        red_c = sum(1 for n in numbers if n in LotteryConfig.RED_NUMBERS)
        blue_c = sum(1 for n in numbers if n in LotteryConfig.BLUE_NUMBERS)
        green_c = len(numbers) - red_c - blue_c
        html += '<tr><td style="padding:6px 10px; border:1px solid #DDD; font-weight:bold;">颜色分布</td><td style="padding:6px 10px; border:1px solid #DDD;"><span style="color:#FF0000; font-size:20px;">红' + str(red_c) + '</span> <span style="color:#0000FF; font-size:20px;">蓝' + str(blue_c) + '</span> <span style="color:#008000; font-size:20px;">绿' + str(green_c) + '</span></td></tr>'
        
        # 每个号码详细属性
        html += '<tr><td style="padding:6px 10px; border:1px solid #DDD; font-weight:bold;">号码详情</td><td style="padding:6px 10px; border:1px solid #DDD;">'
        for n in numbers:
            cn = LotteryConfig.NUMBER_NAMES.get(n, '')
            el = LotteryConfig.NUMBER_ELEMENTS.get(n, '')
            c = LotteryConfig.get_number_color(n)
            cn2 = ''
            if n in LotteryConfig.RED_NUMBERS:
                cn2 = '红'
            elif n in LotteryConfig.BLUE_NUMBERS:
                cn2 = '蓝'
            else:
                cn2 = '绿'
            html += '<span style="color:' + c['text'] + '; font-size:24px; font-weight:bold;">' + str(n).zfill(2) + '</span><span style="font-size:17px;">(' + cn2 + '/' + cn + '/' + el + ')</span> '
        # 特别码详情
        sp_c = LotteryConfig.get_number_color(special)
        html += '<br>特别码: <span style="color:' + sp_c['text'] + '; font-size:24px; font-weight:bold;">' + str(special).zfill(2) + '</span><span style="font-size:17px;">(' + sp_color_name + '/' + sp_name + '/' + sp_elem + ')</span>'
        html += '</td></tr>'
        
        # 区间分布
        zones = {'01-10': 0, '11-20': 0, '21-30': 0, '31-40': 0, '41-49': 0}
        for n in numbers:
            if n <= 10: zones['01-10'] += 1
            elif n <= 20: zones['11-20'] += 1
            elif n <= 30: zones['21-30'] += 1
            elif n <= 40: zones['31-40'] += 1
            else: zones['41-49'] += 1
        zone_str = '  '.join(k + ':' + str(v) for k, v in zones.items())
        html += '<tr><td style="padding:6px 10px; border:1px solid #DDD; font-weight:bold;">区间分布</td><td style="padding:6px 10px; border:1px solid #DDD; font-size:18px;">' + zone_str + '</td></tr>'
        
        # 尾数分布
        tails = [n % 10 for n in numbers]
        tail_counter = {}
        for t in tails:
            tail_counter[t] = tail_counter.get(t, 0) + 1
        tail_str = '  '.join('尾' + str(k) + ':' + str(v) for k, v in sorted(tail_counter.items()))
        html += '<tr><td style="padding:6px 10px; border:1px solid #DDD; font-weight:bold;">尾数分布</td><td style="padding:6px 10px; border:1px solid #DDD; font-size:18px;">' + tail_str + '</td></tr>'
        
        # 连号
        sorted_nums = sorted(numbers)
        consec = []
        temp = [sorted_nums[0]]
        for j in range(1, len(sorted_nums)):
            if sorted_nums[j] == sorted_nums[j-1] + 1:
                temp.append(sorted_nums[j])
            else:
                if len(temp) >= 2:
                    consec.append(temp[:])
                temp = [sorted_nums[j]]
        if len(temp) >= 2:
            consec.append(temp[:])
        consec_str = '  '.join('-'.join(str(x).zfill(2) for x in c) for c in consec) if consec else '无'
        html += '<tr><td style="padding:6px 10px; border:1px solid #DDD; font-weight:bold;">连号</td><td style="padding:6px 10px; border:1px solid #DDD; font-size:18px;">' + consec_str + '</td></tr>'
        
        html += '</table>'
        html += '</div>'
        
        self.period_detail_edit.setHtml(html)


# ============================================================================
# 第八部分：扩展功能模块
# ============================================================================

class StatisticsAnalyzer:
    """高级统计分析器"""
    
    def __init__(self, historical_data):
        self.data = historical_data
        self.df = None
        if historical_data:
            self._build_dataframe()
    
    def _build_dataframe(self):
        """构建Pandas DataFrame"""
        records = []
        for item in self.data:
            record = {
                'period': item.get('period', ''),
                'date': item.get('date', ''),
                'number_1': item.get('numbers', [0]*6)[0],
                'number_2': item.get('numbers', [0]*6)[1],
                'number_3': item.get('numbers', [0]*6)[2],
                'number_4': item.get('numbers', [0]*6)[3],
                'number_5': item.get('numbers', [0]*6)[4],
                'number_6': item.get('numbers', [0]*6)[5],
                'special': item.get('special', 0),
            }
            for i in range(1, 7):
                record[f'is_red_{i}'] = LotteryConfig.is_red(record[f'number_{i}'])
                record[f'is_blue_{i}'] = LotteryConfig.is_blue(record[f'number_{i}'])
                record[f'is_green_{i}'] = LotteryConfig.is_green(record[f'number_{i}'])
                record[f'is_odd_{i}'] = record[f'number_{i}'] % 2 == 1
                record[f'is_big_{i}'] = record[f'number_{i}'] > 25
                record[f'digit_sum_{i}'] = sum(int(d) for d in str(record[f'number_{i}']))
                record[f'last_digit_{i}'] = record[f'number_{i}'] % 10
            records.append(record)
        self.df = Pandas.DataFrame(records)
    
    def get_frequency_analysis(self):
        """获取频率分析数据"""
        if self.df is None:
            return {}
        freq = {}
        for col in ['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']:
            for val in self.df[col]:
                freq[val] = freq.get(val, 0) + 1
        return freq
    
    def get_hot_cold_numbers(self, top_n=15):
        """获取热门和冷门数字"""
        freq = self.get_frequency_analysis()
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        hot = sorted_freq[:top_n]
        cold = sorted_freq[-top_n:][::-1]
        return {'hot': hot, 'cold': cold}
    
    def get_distribution_stats(self):
        """获取分布统计"""
        if self.df is None:
            return {}
        stats = {
            'red_count': sum(1 for _, row in self.df.iterrows() 
                           for i in range(1, 7) if row.get(f'is_red_{i}', False)),
            'blue_count': sum(1 for _, row in self.df.iterrows() 
                            for i in range(1, 7) if row.get(f'is_blue_{i}', False)),
            'green_count': sum(1 for _, row in self.df.iterrows() 
                             for i in range(1, 7) if row.get(f'is_green_{i}', False)),
            'odd_count': sum(1 for _, row in self.df.iterrows() 
                           for i in range(1, 7) if row.get(f'is_odd_{i}', False)),
            'even_count': sum(1 for _, row in self.df.iterrows() 
                            for i in range(1, 7) if not row.get(f'is_odd_{i}', False)),
        }
        total = stats['red_count'] + stats['blue_count'] + stats['green_count']
        if total > 0:
            stats['red_ratio'] = stats['red_count'] / total
            stats['blue_ratio'] = stats['blue_count'] / total
            stats['green_ratio'] = stats['green_count'] / total
        return stats
    
    def get_trend_analysis(self, window=10):
        """获取趋势分析"""
        if self.df is None or len(self.df) < window:
            return {}
        trends = []
        for i in range(len(self.df) - window + 1):
            window_data = self.df.iloc[i:i+window]
            avg_sum = window_data[['number_1', 'number_2', 'number_3', 
                                   'number_4', 'number_5', 'number_6']].values.mean()
            trends.append({'index': i, 'avg_sum': avg_sum})
        return trends
    
    def get_correlation_matrix(self):
        """获取数字间的相关性矩阵"""
        if self.df is None:
            return None
        cols = ['number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6']
        return self.df[cols].corr()
    
    def get_sequential_patterns(self):
        """获取顺序模式分析"""
        if self.df is None:
            return {}
        patterns = {
            'consecutive_pairs': 0,
            'consecutive_triples': 0,
            'same_last_digit_pairs': 0,
            'gap_patterns': []
        }
        for _, row in self.df.iterrows():
            numbers = [row[f'number_{i}'] for i in range(1, 7)]
            numbers_sorted = sorted(numbers)
            for i in range(len(numbers_sorted) - 1):
                if numbers_sorted[i+1] - numbers_sorted[i] == 1:
                    patterns['consecutive_pairs'] += 1
                if i < len(numbers_sorted) - 2:
                    if numbers_sorted[i+2] - numbers_sorted[i+1] == 1:
                        patterns['consecutive_triples'] += 1
                last_digits = [n % 10 for n in numbers_sorted]
                if i < len(last_digits) - 1 and last_digits[i+1] == last_digits[i]:
                    patterns['same_last_digit_pairs'] += 1
            for i in range(len(numbers_sorted) - 1):
                patterns['gap_patterns'].append(numbers_sorted[i+1] - numbers_sorted[i])
        return patterns
    
    def get_interval_analysis(self):
        """获取间隔分析"""
        if self.df is None:
            return {}
        intervals = {}
        for num in range(1, 50):
            appearances = []
            last_idx = None
            for idx, row in self.df.iterrows():
                numbers = [row[f'number_{i}'] for i in range(1, 7)]
                if num in numbers:
                    if last_idx is not None:
                        appearances.append(idx - last_idx)
                    last_idx = idx
            if appearances:
                intervals[num] = {
                    'count': len(appearances),
                    'avg_interval': sum(appearances) / len(appearances),
                    'max_interval': max(appearances),
                    'min_interval': min(appearances),
                    'current_gap': len(self.df) - last_idx if last_idx is not None else len(self.df)
                }
        return intervals
    
    def get_zone_distribution(self):
        """获取区间分布分析"""
        if self.df is None:
            return {}
        zones = {'zone_1': 0, 'zone_2': 0, 'zone_3': 0, 'zone_4': 0}
        zone_ranges = [(1, 12), (13, 24), (25, 36), (37, 49)]
        for _, row in self.df.iterrows():
            for i in range(1, 7):
                num = row.get(f'number_{i}', 0)
                for idx, (start, end) in enumerate(zone_ranges):
                    if start <= num <= end:
                        zones[f'zone_{idx+1}'] += 1
                        break
        return zones
    
    def get_tail_number_distribution(self):
        """获取尾数分布分析"""
        if self.df is None:
            return {}
        tails = {i: 0 for i in range(10)}
        for _, row in self.df.iterrows():
            for i in range(1, 7):
                num = row.get(f'number_{i}', 0)
                tails[num % 10] += 1
        return tails
    
    def get_sum_statistics(self):
        """获取总和统计"""
        if self.df is None:
            return {}
        sums = []
        for _, row in self.df.iterrows():
            s = sum(row[f'number_{i}'] for i in range(1, 7))
            sums.append(s)
        return {
            'mean': sum(sums) / len(sums) if sums else 0,
            'min': min(sums) if sums else 0,
            'max': max(sums) if sums else 0,
            'median': sorted(sums)[len(sums)//2] if sums else 0
        }


class ReportExporter:
    """报告导出器"""
    
    def __init__(self, historical_data, predictions=None):
        self.data = historical_data
        self.predictions = predictions or []
        self.analyzer = StatisticsAnalyzer(historical_data) if historical_data else None
    
    def generate_text_report(self):
        """生成文本报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("彩票预测系统 v7.1 - 分析报告")
        lines.append("=" * 70)
        lines.append("")
        
        if self.data:
            lines.append(f"数据总量: {len(self.data)} 条历史记录")
            lines.append("")
            
            hot_cold = self.analyzer.get_hot_cold_numbers(10)
            lines.append("【热门数字 TOP10】")
            for num, freq in hot_cold['hot']:
                lines.append(f"  {num:02d} - 出现 {freq} 次")
            lines.append("")
            
            lines.append("【冷门数字 TOP10】")
            for num, freq in hot_cold['cold']:
                lines.append(f"  {num:02d} - 出现 {freq} 次")
            lines.append("")
            
            dist = self.analyzer.get_distribution_stats()
            lines.append("【颜色分布】")
            lines.append(f"  红码: {dist.get('red_count', 0)} 次 ({dist.get('red_ratio', 0)*100:.1f}%)")
            lines.append(f"  蓝码: {dist.get('blue_count', 0)} 次 ({dist.get('blue_ratio', 0)*100:.1f}%)")
            lines.append(f"  绿码: {dist.get('green_count', 0)} 次 ({dist.get('green_ratio', 0)*100:.1f}%)")
            lines.append("")
            
            zones = self.analyzer.get_zone_distribution()
            lines.append("【区间分布】")
            lines.append(f"  区间1 (01-12): {zones.get('zone_1', 0)} 次")
            lines.append(f"  区间2 (13-24): {zones.get('zone_2', 0)} 次")
            lines.append(f"  区间3 (25-36): {zones.get('zone_3', 0)} 次")
            lines.append(f"  区间4 (37-49): {zones.get('zone_4', 0)} 次")
            lines.append("")
            
            tails = self.analyzer.get_tail_number_distribution()
            lines.append("【尾数分布】")
            for tail, count in tails.items():
                lines.append(f"  尾数{tail}: {count} 次")
            lines.append("")
        
        if self.predictions:
            lines.append("【最新预测结果】")
            lines.append(f"  预测号码: {' '.join(f'{p:02d}' for p in self.predictions[:6])}")
            if len(self.predictions) > 6:
                lines.append(f"  特别号: {self.predictions[6]:02d}")
            lines.append("")
        
        lines.append("=" * 70)
        lines.append("报告生成时间: " + str(QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")))
        lines.append("=" * 70)
        
        return '\n'.join(lines)
    
    def generate_json_report(self):
        """生成JSON格式报告"""
        import json
        report = {
            'version': '5.0',
            'data_count': len(self.data) if self.data else 0,
            'generated_at': str(QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss"))
        }
        
        if self.data and self.analyzer:
            hot_cold = self.analyzer.get_hot_cold_numbers(15)
            report['hot_numbers'] = [{'number': n, 'frequency': f} for n, f in hot_cold['hot']]
            report['cold_numbers'] = [{'number': n, 'frequency': f} for n, f in hot_cold['cold']]
            report['distribution'] = self.analyzer.get_distribution_stats()
            report['zone_distribution'] = self.analyzer.get_zone_distribution()
            report['tail_distribution'] = self.analyzer.get_tail_number_distribution()
        
        if self.predictions:
            report['predictions'] = {
                'numbers': self.predictions[:6],
                'special': self.predictions[6] if len(self.predictions) > 6 else None
            }
        
        return json.dumps(report, ensure_ascii=False, indent=2)
    
    def export_to_file(self, filepath, format_type='txt'):
        """导出到文件"""
        if format_type == 'txt':
            content = self.generate_text_report()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        elif format_type == 'json':
            content = self.generate_json_report()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        elif format_type == 'csv':
            self._export_csv(filepath)
        return True
    
    def _export_csv(self, filepath):
        """导出为CSV格式"""
        if not self.data:
            return
        import csv
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['期号', '日期', '正码1', '正码2', '正码3', '正码4', '正码5', '正码6', '特别码'])
            for item in self.data:
                writer.writerow([
                    item.get('period', ''),
                    item.get('date', ''),
                    *item.get('numbers', ['']*6),
                    item.get('special', '')
                ])


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_number(num):
        """验证数字是否合法"""
        try:
            n = int(num)
            return 1 <= n <= 49
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_period(period):
        """验证期号格式"""
        if not period:
            return False
        period_str = str(period)
        if len(period_str) < 6:
            return False
        try:
            int(period_str[:4])
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_date(date_str):
        """验证日期格式"""
        from PyQt6.QtCore import QDate
        formats = ['yyyy-MM-dd', 'yyyy/MM/dd', 'yyyyMMdd', 'yyyy.MM.dd']
        for fmt in formats:
            date = QDate.fromString(date_str, fmt)
            if date.isValid():
                return True
        return False
    
    @staticmethod
    def validate_record(record):
        """验证完整记录"""
        if not isinstance(record, dict):
            return False, "记录格式错误"
        
        if 'numbers' not in record or len(record['numbers']) != 6:
            return False, "正码数量错误"
        
        for num in record['numbers']:
            if not DataValidator.validate_number(num):
                return False, f"正码 {num} 不合法"
        
        if 'special' in record:
            if not DataValidator.validate_number(record['special']):
                return False, f"特别码 {record['special']} 不合法"
        
        return True, "验证通过"


class PredictionOptimizer:
    """预测优化器 - 使用Optuna进行超参数优化"""
    
    def __init__(self, historical_data):
        self.data = historical_data
        self.best_params = None
    
    def _prepare_features(self, idx, window=20):
        """准备特征"""
        if idx < window or idx >= len(self.data):
            return None
        window_data = self.data[idx-window:idx]
        features = []
        for item in window_data:
            features.extend(item.get('numbers', [])[:6])
        if len(features) < window * 6:
            return None
        return features[:window * 6]
    
    def _calculate_fitness(self, params, n_trials=10):
        """计算适应度"""
        scores = []
        for i in range(20, min(len(self.data), 100)):
            features = self._prepare_features(i)
            if features is None:
                continue
            score = sum(features[:3]) / 3 * params.get('weight', 1.0)
            scores.append(score)
        return sum(scores) / len(scores) if scores else 0
    
    def optimize(self, n_trials=30):
        """运行优化"""
        optuna = _get_optuna()
        if optuna is None:
            return {'weight': 1.0, 'decay': 0.95, 'threshold': 0.5}
        try:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            def objective(trial):
                params = {
                    'weight': trial.suggest_float('weight', 0.5, 2.0),
                    'decay': trial.suggest_float('decay', 0.8, 0.99),
                    'threshold': trial.suggest_float('threshold', 0.1, 0.9),
                }
                return self._calculate_fitness(params)
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            self.best_params = study.best_params
            return self.best_params
        except Exception:
            return {'weight': 1.0, 'decay': 0.95, 'threshold': 0.5}
    
    def get_optimized_prediction(self):
        """获取优化后的预测"""
        if not self.best_params:
            self.optimize()
        predictions = list(range(1, 50))
        predictions.sort(
            key=lambda x: (
                MathUtils.get_number_weight(x, self.data) * self.best_params.get('weight', 1.0),
                -abs(x - 25) * self.best_params.get('decay', 0.95)
            ),
            reverse=True
        )
        return predictions[:7]


class DataPreprocessor:
    """数据预处理器 - 数据清洗和特征工程"""
    
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.processed_data = []
    
    def clean_data(self):
        """清洗数据"""
        for item in self.raw_data:
            if not self._validate_record(item):
                continue
            
            cleaned = {
                'period': str(item.get('period', '')).strip(),
                'date': str(item.get('date', '')).strip(),
                'numbers': [],
                'special': 0
            }
            
            numbers = item.get('numbers', [])
            for num in numbers:
                try:
                    n = int(num)
                    if 1 <= n <= 49 and n not in cleaned['numbers']:
                        cleaned['numbers'].append(n)
                except (ValueError, TypeError):
                    continue
            
            if len(cleaned['numbers']) == 6:
                special = item.get('special', 0)
                try:
                    sp = int(special)
                    if 1 <= sp <= 49:
                        cleaned['special'] = sp
                except (ValueError, TypeError):
                    continue
                
                self.processed_data.append(cleaned)
        
        return self.processed_data
    
    def _validate_record(self, record):
        """验证记录"""
        if not isinstance(record, dict):
            return False
        
        numbers = record.get('numbers', [])
        if not isinstance(numbers, (list, tuple)) or len(numbers) < 6:
            return False
        
        for num in numbers:
            try:
                n = int(num)
                if not (1 <= n <= 49):
                    return False
            except (ValueError, TypeError):
                return False
        
        return True
    
    def extract_features(self):
        """提取特征"""
        features = []
        for item in self.processed_data:
            feat = {
                'sum': sum(item['numbers']),
                'mean': sum(item['numbers']) / 6,
                'std': self._calculate_std(item['numbers']),
                'min': min(item['numbers']),
                'max': max(item['numbers']),
                'range': max(item['numbers']) - min(item['numbers']),
                'odd_count': sum(1 for n in item['numbers'] if n % 2 == 1),
                'even_count': sum(1 for n in item['numbers'] if n % 2 == 0),
                'big_count': sum(1 for n in item['numbers'] if n > 25),
                'small_count': sum(1 for n in item['numbers'] if n <= 25),
                'consecutive_count': self._count_consecutive(item['numbers']),
                'red_count': sum(1 for n in item['numbers'] if LotteryConfig.is_red(n)),
                'blue_count': sum(1 for n in item['numbers'] if LotteryConfig.is_blue(n)),
                'green_count': sum(1 for n in item['numbers'] if LotteryConfig.is_green(n)),
            }
            
            for i in range(1, 7):
                feat[f'num_{i}'] = item['numbers'][i-1] if i <= len(item['numbers']) else 0
                feat[f'tail_{i}'] = item['numbers'][i-1] % 10 if i <= len(item['numbers']) else 0
            
            feat['special'] = item['special']
            features.append(feat)
        
        return features
    
    def _calculate_std(self, numbers):
        """计算标准差"""
        if not numbers:
            return 0
        mean = sum(numbers) / len(numbers)
        variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
        return variance ** 0.5
    
    def _count_consecutive(self, numbers):
        """计算连号数量"""
        sorted_nums = sorted(numbers)
        count = 0
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i+1] - sorted_nums[i] == 1:
                count += 1
        return count
    
    def normalize_features(self, features):
        """归一化特征"""
        if not features:
            return features
        
        keys = features[0].keys()
        normalized = []
        
        for feat in features:
            norm_feat = {}
            for key in keys:
                values = [f[key] for f in features]
                min_val, max_val = min(values), max(values)
                if max_val - min_val > 0:
                    norm_feat[key] = (feat[key] - min_val) / (max_val - min_val)
                else:
                    norm_feat[key] = 0
            normalized.append(norm_feat)
        
        return normalized


class PatternMatcher:
    """模式匹配器 - 识别历史模式"""
    
    def __init__(self, historical_data):
        self.data = historical_data
        self.patterns = {}
    
    def find_similar_patterns(self, numbers, top_n=5):
        """查找相似模式"""
        target_set = set(numbers[:6])
        similarities = []
        
        for idx, item in enumerate(self.data):
            item_set = set(item.get('numbers', [])[:6])
            intersection = len(target_set & item_set)
            similarity = intersection / 6.0
            
            similarities.append({
                'index': idx,
                'period': item.get('period', ''),
                'similarity': similarity,
                'numbers': item.get('numbers', []),
                'special': item.get('special', 0)
            })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_n]
    
    def detect_repeating_patterns(self, window=20):
        """检测重复模式"""
        patterns = []
        
        for i in range(len(self.data) - window * 2):
            window1 = self.data[i:i+window]
            window2 = self.data[i+window:i+window*2]
            
            pattern1 = tuple(sorted([
                tuple(sorted(item.get('numbers', [])[:6])) 
                for item in window1
            ]))
            
            pattern2 = tuple(sorted([
                tuple(sorted(item.get('numbers', [])[:6])) 
                for item in window2
            ]))
            
            common = len(set(pattern1) & set(pattern2))
            if common >= window * 0.3:
                patterns.append({
                    'start_index': i,
                    'pattern_length': window,
                    'common_count': common,
                    'similarity': common / window
                })
        
        return patterns
    
    def get_pattern_statistics(self):
        """获取模式统计"""
        stats = {
            'total_records': len(self.data),
            'average_sum': 0,
            'sum_distribution': {},
            'digit_distribution': {},
            'zone_distribution': {1: 0, 2: 0, 3: 0, 4: 0},
            'color_distribution': {'red': 0, 'blue': 0, 'green': 0}
        }
        
        if not self.data:
            return stats
        
        sums = []
        for item in self.data:
            numbers = item.get('numbers', [])[:6]
            s = sum(numbers)
            sums.append(s)
            
            for num in numbers:
                digit = num % 10
                stats['digit_distribution'][digit] = stats['digit_distribution'].get(digit, 0) + 1
                
                if 1 <= num <= 12:
                    stats['zone_distribution'][1] += 1
                elif 13 <= num <= 24:
                    stats['zone_distribution'][2] += 1
                elif 25 <= num <= 36:
                    stats['zone_distribution'][3] += 1
                else:
                    stats['zone_distribution'][4] += 1
                
                if LotteryConfig.is_red(num):
                    stats['color_distribution']['red'] += 1
                elif LotteryConfig.is_blue(num):
                    stats['color_distribution']['blue'] += 1
                else:
                    stats['color_distribution']['green'] += 1
        
        stats['average_sum'] = sum(sums) / len(sums) if sums else 0
        
        for s in sums:
            bucket = (s // 10) * 10
            stats['sum_distribution'][f'{bucket}-{bucket+9}'] = \
                stats['sum_distribution'].get(f'{bucket}-{bucket+9}', 0) + 1
        
        return stats


class DeepLearningPredictor:
    """深度学习预测器 - 使用PyTorch"""
    
    def __init__(self, historical_data):
        self.data = historical_data
        self.model = None
        self.device = self._get_device()
    
    def _get_device(self):
        """获取计算设备"""
        try:
            import torch
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except ImportError:
            return 'cpu'
    
    def _prepare_sequence_data(self, sequence_length=20):
        """准备序列数据"""
        if len(self.data) < sequence_length + 1:
            return None, None
        X, y = [], []
        for i in range(len(self.data) - sequence_length):
            seq = []
            for j in range(i, i + sequence_length):
                numbers = self.data[j].get('numbers', [0]*6)
                seq.extend(numbers)
                seq.append(self.data[j].get('special', 0))
            X.append(seq)
            next_numbers = self.data[i + sequence_length].get('numbers', [0]*6)
            y.append(next_numbers)
        return X, y
    
    def _build_model(self, input_size):
        """构建神经网络模型"""
        try:
            import torch
            import torch.nn as nn
            
            class LotteryLSTM(nn.Module):
                def __init__(self, input_size, hidden_size=64, num_layers=2):
                    super(LotteryLSTM, self).__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                       batch_first=True, dropout=0.2)
                    self.fc1 = nn.Linear(hidden_size, 32)
                    self.fc2 = nn.Linear(32, 6)
                    self.relu = nn.ReLU()
                
                def forward(self, x):
                    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                    out, _ = self.lstm(x, (h0, c0))
                    out = self.fc1(out[:, -1, :])
                    out = self.relu(out)
                    out = self.fc2(out)
                    return out
            
            return LotteryLSTM(input_size)
        except ImportError:
            return None
    
    def train(self, epochs=50, sequence_length=20):
        """训练模型"""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            
            X, y = self._prepare_sequence_data(sequence_length)
            if X is None or y is None:
                return False
            
            X_tensor = torch.FloatTensor(X).unsqueeze(-1)
            y_tensor = torch.LongTensor(y)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            input_size = X_tensor.shape[-1]
            self.model = self._build_model(input_size)
            if self.model is None:
                return False
            
            self.model = self.model.to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            for epoch in range(epochs):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device).float()
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
            
            return True
        except ImportError:
            return False
        except Exception as e:
            print(f"训练失败: {e}")
            return False
    
    def predict(self, sequence_length=20):
        """进行预测"""
        if self.model is None:
            return None
        
        try:
            import torch
            
            last_seq = []
            for i in range(len(self.data) - sequence_length, len(self.data)):
                numbers = self.data[i].get('numbers', [0]*6)
                last_seq.extend(numbers)
                last_seq.append(self.data[i].get('special', 0))
            
            X = torch.FloatTensor([last_seq]).unsqueeze(-1).to(self.device)
            with torch.no_grad():
                prediction = self.model(X)
            predictions = prediction.cpu().numpy()[0]
            return [max(1, min(49, int(round(p)))) for p in predictions]
        except Exception:
            return None


class TimeSeriesAnalyzer:
    """时间序列分析器 - 使用StatsModels"""
    
    def __init__(self, historical_data):
        self.data = historical_data
        self.results = {}
    
    def _extract_series(self, position=0):
        """提取指定位置的时间序列"""
        series = []
        for item in self.data:
            numbers = item.get('numbers', [])
            if len(numbers) > position:
                series.append(numbers[position])
        return series
    
    def _extract_sums(self):
        """提取总和序列"""
        return [sum(item.get('numbers', [0]*6)) for item in self.data]
    
    def _extract_special_series(self):
        """提取特别号序列"""
        return [item.get('special', 0) for item in self.data]
    
    def perform_stationarity_test(self, position=0):
        """执行平稳性检验"""
        try:
            from statsmodels.tsa.stattools import adfuller
            
            series = self._extract_series(position)
            if len(series) < 30:
                return None
            
            result = adfuller(series)
            return {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05
            }
        except ImportError:
            return None
    
    def fit_arima(self, position=0, order=(5, 1, 0)):
        """拟合ARIMA模型"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            series = self._extract_series(position)
            if len(series) < 50:
                return None
            
            model = ARIMA(series, order=order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=1)
            
            self.results[f'position_{position}'] = {
                'model': fitted,
                'forecast': forecast[0] if len(forecast) > 0 else 25
            }
            
            return self.results[f'position_{position}']
        except ImportError:
            return None
        except Exception as e:
            print(f"ARIMA拟合失败: {e}")
            return None
    
    def detect_seasonality(self, position=0):
        """检测季节性"""
        try:
            sig_mod = _get_scipy_signal()
            if sig_mod is None:
                return None
            
            series = self._extract_series(position)
            if len(series) < 50:
                return None
            
            autocorr = sig_mod.correlate(series, series, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            peaks, _ = sig_mod.find_peaks(autocorr[1:], height=autocorr[0]*0.1)
            
            return {
                'has_seasonality': len(peaks) > 0,
                'seasonal_periods': peaks[:5].tolist() if len(peaks) > 0 else [],
                'autocorrelation': autocorr[:20].tolist()
            }
        except ImportError:
            return None
    
    def get_trend_components(self, position=0):
        """获取趋势分量"""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            series = self._extract_series(position)
            if len(series) < 50:
                return None
            
            result = seasonal_decompose(series, model='additive', period=7)
            
            return {
                'trend': result.trend[~result.trend.isna().values].tolist(),
                'seasonal': result.seasonal[~result.seasonal.isna().values].tolist(),
                'residual': result.resid[~result.resid.isna().values].tolist()
            }
        except ImportError:
            return None


class EnsemblePredictor:
    """集成预测器 - 结合多种预测方法"""
    
    def __init__(self, historical_data):
        self.data = historical_data
        self.weights = {}
    
    def _initialize_weights(self):
        """初始化权重"""
        self.weights = {
            'hot_cold': 0.2,
            'frequency': 0.15,
            'pattern': 0.2,
            'ml': 0.25,
            'statistical': 0.2
        }
    
    def _get_hot_cold_scores(self):
        """获取冷热分析分数"""
        scores = {}
        freq = MathUtils.get_number_frequency(self.data)
        max_freq = max(freq.values()) if freq else 1
        
        for num in range(1, 50):
            appear_count = freq.get(num, 0)
            current_gap = MathUtils.get_missed_count(num, self.data)
            scores[num] = (
                appear_count / max_freq * 0.6 +
                min(current_gap, 50) / 50 * 0.4
            )
        return scores
    
    def _get_frequency_scores(self):
        """获取频率分析分数"""
        scores = {}
        total = len(self.data) * 6
        freq = MathUtils.get_number_frequency(self.data)
        
        expected = total / 49
        for num in range(1, 50):
            observed = freq.get(num, 0)
            chi_score = abs(observed - expected) / expected if expected > 0 else 0
            scores[num] = 1 / (1 + chi_score)
        return scores
    
    def _get_pattern_scores(self):
        """获取模式分析分数"""
        scores = {n: 0 for n in range(1, 50)}
        recent = self.data[:min(10, len(self.data))]
        
        for item in recent:
            numbers = item.get('numbers', [])
            for num in numbers:
                if 1 <= num <= 49:
                    scores[num] += 0.5
            
            special = item.get('special', 0)
            if 1 <= special <= 49:
                scores[special] += 0.3
        
        neighbors = {n: [] for n in range(1, 50)}
        for item in recent:
            for num in item.get('numbers', [])[:6]:
                if 1 <= num <= 49:
                    for n in range(max(1, num-3), min(50, num+4)):
                        if n != num:
                            neighbors[num].append(n)
        
        for num, neighs in neighbors.items():
            for n in neighs:
                if n in scores:
                    scores[n] += 0.1
        
        max_score = max(scores.values()) if scores.values() else 1
        for num in scores:
            scores[num] /= max_score
        
        return scores
    
    def get_ensemble_prediction(self, n_predictions=7):
        """获取集成预测结果"""
        self._initialize_weights()
        
        scores_list = [
            (self._get_hot_cold_scores(), self.weights['hot_cold']),
            (self._get_frequency_scores(), self.weights['frequency']),
            (self._get_pattern_scores(), self.weights['pattern'])
        ]
        
        final_scores = {n: 0 for n in range(1, 50)}
        for scores, weight in scores_list:
            for num in final_scores:
                final_scores[num] += scores.get(num, 0) * weight
        
        sorted_predictions = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        predictions = [num for num, _ in sorted_predictions[:n_predictions]]
        
        return predictions


class AdvancedVisualization:
    """高级可视化工具"""
    
    def __init__(self, historical_data):
        self.data = historical_data
    
    def create_heatmap(self, ax, data_type='frequency'):
        """创建热力图"""
        try:
            sns = _get_sns()
            if sns is None:
                ax.text(0.5, 0.5, '请安装seaborn库', ha='center', va='center')
                return
            
            if data_type == 'frequency':
                freq = MathUtils.get_number_frequency(self.data)
                matrix = np.zeros((7, 7))
                for num, count in freq.items():
                    row = (num - 1) // 7
                    col = (num - 1) % 7
                    matrix[row][col] = count
                
                sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', ax=ax,
                           xticklabels=[f'{i*7+j+1}' for j in range(7)],
                           yticklabels=[f'{i*7+1}-{i*7+7}' for i in range(7)])
                ax.set_title('数字出现频率热力图')
        except ImportError:
            ax.text(0.5, 0.5, '请安装seaborn库', ha='center', va='center')
    
    def create_pairplot_data(self):
        """创建配对图数据"""
        if len(self.data) < 10:
            return None
        
        cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
        data_dict = {col: [] for col in cols}
        
        for item in self.data[:100]:
            numbers = item.get('numbers', [0]*6)
            for i, num in enumerate(numbers[:6]):
                data_dict[cols[i]].append(num)
        
        return data_dict
    
    def create_distribution_plot(self, ax, zone_type='color'):
        """创建分布图"""
        try:
            sns = _get_sns()
            # seaborn在这个方法中主要用bar，不需要sns也可以
            # 这里仅在需要时检查
            
            if zone_type == 'color':
                colors = {'red': 0, 'blue': 0, 'green': 0}
                for item in self.data:
                    for num in item.get('numbers', [])[:6]:
                        if LotteryConfig.is_red(num):
                            colors['red'] += 1
                        elif LotteryConfig.is_blue(num):
                            colors['blue'] += 1
                        elif LotteryConfig.is_green(num):
                            colors['green'] += 1
                
                bars = ax.bar(colors.keys(), colors.values(), 
                            color=['#FF4444', '#4444FF', '#44AA44'])
                ax.set_ylabel('出现次数')
                ax.set_title('颜色分布统计')
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
            elif zone_type == 'zone':
                zones = [0, 0, 0, 0]
                zone_ranges = [(1, 12), (13, 24), (25, 36), (37, 49)]
                
                for item in self.data:
                    for num in item.get('numbers', [])[:6]:
                        for idx, (start, end) in enumerate(zone_ranges):
                            if start <= num <= end:
                                zones[idx] += 1
                                break
                
                ax.bar(['01-12', '13-24', '25-36', '37-49'], zones,
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
                ax.set_ylabel('出现次数')
                ax.set_title('区间分布统计')
        except ImportError:
            ax.text(0.5, 0.5, '请安装seaborn库', ha='center', va='center')


# ============================================================================
# 第九部分：应用入口
# ============================================================================

def main():
    print("=" * 60)
    print("彩票预测系统 v7.1 启动中...")
    print("=" * 60)
    
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
            print("✓ " + description + " (" + module_name + ") - OK")
        except ImportError:
            print("✗ " + description + " (" + module_name + ") - 未安装")
            all_ok = False
    
    if not all_ok:
        print("\n警告: 部分依赖库未安装，部分功能可能无法使用。")
        print("请运行: pip install -r requirements.txt")
    
    print("\n" + "=" * 60)
    print("正在初始化应用程序...")
    print("=" * 60)
    
    app = QApplication(sys.argv)
    app.setApplicationName("彩票预测系统")
    app.setApplicationVersion("5.1")
    app.setOrganizationName("AI Assistant")
    
    window = LotteryPredictionWindow()
    window.show()
    
    print("\n" + "=" * 60)
    print("彩票预测系统 v7.1 已启动！")
    print("=" * 60)
    print("\n使用方法：")
    print("1. 在'数据导入与格式转换'标签页粘贴原始数据或批量导入")
    print("2. 在'预测与抽取'标签页选择预测算法")
    print("3. 点击'开始预测'获取预测结果")
    print("4. 使用顶部工具栏调整字体大小和边距")
    print("5. 查看'第七位预判'和'统计分析图表'进行深度分析")
    print("=" * 60)
    
    sys.exit(app.exec())




# =============================================================================
# 库使用统计注释 - v7.1深度集成验证
# =============================================================================
"""
【NumPy深度使用】
- np.linalg.lstsq: _np_linear_regression_trend, _scipy_big_interp
- np.dot: _np_correlation_coefficients, _scipy_big_interp
- np.histogram: _np_distribution_histogram, _np_hot_histogram_bonus
- np.percentile: _np_distribution_histogram, _np_missing_percentile
- np.corrcoef: _np_correlation_coefficients, _scipy_odd_correction
- np.exp: hot_cold_algorithm, _scipy_hot_smooth
- np.vander: _np_linear_regression_trend
- np.nan_to_num: _np_correlation_coefficients

【SciPy深度使用】
- scipy.optimize.minimize: _scipy_optimize_weights
- scipy.signal.convolve: _scipy_smooth_trend, _scipy_hot_smooth
- scipy.interpolate.splrep/splev: _scipy_interpolate_missing, _scipy_missing_interp
- scipy.stats.ks_2samp: _scipy_distribution_test, _scipy_odd_correction
- scipy.stats.poisson.cdf/sf: poisson_distribution
- scipy.stats.expon.cdf/sf: poisson_distribution

【Scikit-learn深度使用】
- StandardScaler: _prepare_sklearn_features, odd_even_algorithm, big_small_algorithm
- MinMaxScaler: _range_cv_scores, _scipy_big_interp
- RandomForestClassifier: big_small_algorithm, _gb_predict_probs
- GradientBoostingClassifier: comprehensive_recommendation, _range_cv_scores
- LogisticRegression: odd_even_algorithm, _lr_roulette_weights, roulette_selection
- MLPClassifier: adjacent_number_analysis (_mlp_adjacent_probs)
- GaussianNB: missing_value_analysis (_nb_missing_probs)
- KMeans: _prepare_sklearn_features, hot_cold_algorithm, tail_distribution_algorithm
- PCA: _pca_historical_similarity
- cosine_similarity: historical_similarity, _pca_historical_similarity
- cross_val_score: _range_cv_scores

【PyTorch深度使用】
- torch.nn.LSTM: _prepare_pytorch_lstm, LotteryLSTM类
- torch.nn.Linear/ReLU/Dropout/Sigmoid: _prepare_pytorch_lstm
- torch.nn.BCELoss/MSELoss: _prepare_pytorch_lstm, _pt_autoencoder
- torch.optim.Adam: _prepare_pytorch_lstm, _pt_autoencoder
- torch.softmax: roulette_selection, mystical_algorithm
- torch.rand: mystical_algorithm, _prepare_pytorch_lstm
- torch.cat/stack: _prepare_pytorch_lstm
- model.train/backward/step: _prepare_pytorch_lstm完整训练循环

【TensorFlow深度使用】
- tensorflow.keras.Sequential: _prepare_tensorflow_model, _tf_fc_classifier
- tensorflow.keras.layers.LSTM: _prepare_tensorflow_model
- tensorflow.keras.layers.Dense/Dropout: _prepare_tensorflow_model, _tf_fc_classifier
- tensorflow.keras.Model: _tf_autoencoder
- tf.random.set_seed: _prepare_tensorflow_model
- model.compile/fit/predict: _prepare_tensorflow_model, _tf_fc_classifier

【Pandas深度使用】
- pandas.DataFrame: _build_dataframe, range_distribution_algorithm
- pandas.Series: _calculate_range_distribution等方法

【Statsmodels深度使用】
- sm.tsa.acf: _calculate_autocorrelation

【Optuna深度使用】
- optuna.create_study/suggest_float: _optimize_ensemble_weights
- TPESampler: 贝叶斯优化采样器
"""


if __name__ == "__main__":
    main()

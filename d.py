#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
数据分析预测系统 (Data Analysis & Prediction System)
================================================================================
一个基于PyQt6开发的完整数据分析与预测系统

主要功能：
1. 数据转换面板 - 粘贴并转换各种格式的数据
2. 分析预测面板 - 49个数字随机抽取与多种预测算法

技术栈：PyQt6, NumPy, Pandas, Matplotlib, Seaborn, Scipy, Statsmodels,
       Scikit-learn, Optuna, torch

作者：AI Assistant
版本：1.0.0
================================================================================
"""

# ============================================================================
# 导入必要的库
# ============================================================================
import sys
import os
import re
import json
import random
import hashlib
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Callable
from collections import Counter
import math

# PyQt6 相关导入
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QTextEdit, QLineEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QSlider, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QSplitter, QSplitterHandle, QStyle, QStyleOptionSlider,
    QToolBar, QStatusBar, QMenuBar, QMenu, QDialog, QMessageBox,
    QFrame, QSizePolicy, QScrollBar, QCheckBox, QListWidget,
    QListWidgetItem, QAbstractItemView, QProgressBar, QToolButton,
    QFontComboBox, QColorDialog, QFileDialog, QInputDialog
)
from PyQt6.QtCore import (
    Qt, QSize, QPoint, QRect, QTimer, QThread, pyqtSignal,
    QObject, QPropertyAnimation, QEasingCurve, QSettings, QFile,
    QTextStream, QSaveFile
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QPainter, QBrush, QPen, QIcon,
    QAction, QKeySequence, QCursor, QScreen, QFontDatabase,
    QTextCursor, QTextCharFormat, QTextBlockFormat, QRegularExpressionValidator
)

# 数据科学库导入（带优雅降级）
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import matplotlib

    matplotlib.use('Agg')  # 使用非GUI后端
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

try:
    from scipy import stats
    from scipy.special import comb, perm, gamma

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None

try:
    import statsmodels.api as sm

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    sm = None

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    RandomForestClassifier = None

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None


# ============================================================================
# 全局配置与常量
# ============================================================================

class Config:
    """全局配置类 - 管理所有可配置参数"""

    # 窗口配置
    WINDOW_TITLE = "数据分析预测系统 v1.0"
    WINDOW_MIN_WIDTH = 1200
    WINDOW_MIN_HEIGHT = 800

    # 配色方案 - 纯色定义（无暗色系）
    COLORS = {
        # 主色调
        'primary': '#2196F3',  # 蓝色
        'primary_dark': '#1976D2',  # 深蓝
        'primary_light': '#BBDEFB',  # 浅蓝

        # 成功/失败色
        'success': '#4CAF50',  # 成功绿
        'success_light': '#C8E6C9',  # 浅成功绿
        'danger': '#F44336',  # 失败红
        'danger_light': '#FFCDD2',  # 浅失败红

        # 功能色
        'warning': '#FF9800',  # 警告橙
        'info': '#00BCD4',  # 信息青
        'purple': '#9C27B0',  # 紫色
        'pink': '#E91E63',  # 粉色
        'indigo': '#3F51B5',  # 靛蓝

        # 中性色（仅用于特定场景）
        'white': '#FFFFFF',  # 纯白
        'text': '#000000',  # 纯黑文字
        'text_secondary': '#424242',  # 深灰文字
        'border': '#BDBDBD',  # 边框灰
        'highlight': '#FFF176',  # 高亮黄
    }

    # 背景色
    BACKGROUND = '#FFFFFF'
    SECONDARY_BG = '#FFF8E1'  # 浅黄背景（仅用于输入区域）

    # 字体大小映射（初号到小四）
    FONT_SIZES = {
        '初号': 42,
        '小初': 36,
        '一号': 26,
        '小一': 24,
        '二号': 22,
        '小二': 18,
        '三号': 16,
        '小三': 15,
        '四号': 14,
        '小四': 12,
    }

    # 默认字体大小索引
    DEFAULT_FONT_INDEX = 8  # 小四

    # 间距配置
    MARGIN_SMALL = 5
    MARGIN_MEDIUM = 10
    MARGIN_LARGE = 15
    MARGIN_XLARGE = 20

    # 边距配置
    PADDING_SMALL = 5
    PADDING_MEDIUM = 10
    PADDING_LARGE = 15

    # 数字范围
    MIN_NUMBER = 1
    MAX_NUMBER = 49
    NUMBERS_PER_ROW = 7  # 每行显示的数字个数

    # 数据文件路径
    DATA_FILE = 'analysis_data.json'

    # 算法列表
    ALGORITHMS = [
        '综合推荐',
        '冷热数字算法',
        '单双算法',
        '大小算法',
        '遗漏值分析算法',
        '连号/邻号分析算法',
        '尾数分布算法',
        '区间分布算法',
        '轮盘赌选择算法',
        '历史相似性算法',
        '泊松概率分布算法',
        '玄学算法',
    ]


# ============================================================================
# 工具函数
# ============================================================================

def hash_string(s: str) -> str:
    """字符串哈希"""
    return hashlib.md5(s.encode()).hexdigest()[:8]


def parse_lottery_data(text: str) -> Optional[Dict[str, Any]]:
    """
    解析彩票数据
    支持格式：
    - 第116期最新开奖结果 2026年04月26日 15 龙/水 46 鸡/木 16 兔/木 10 鸡/火 48 羊/火 33 狗/火 22 鸡/水
    - 第116期 15 46 16 10 48 33+22
    """
    text = text.strip()

    # 提取期号
    period_match = re.search(r'第\s*(\d+)\s*期', text)
    if not period_match:
        return None
    period = period_match.group(1)

    # 提取日期
    date_match = re.search(r'(\d{4})[年\-\.](\d{1,2})[月\-\.](\d{1,2})', text)
    if date_match:
        date_str = f"{date_match.group(1)}-{date_match.group(2).zfill(2)}-{date_match.group(3).zfill(2)}"
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')

    # 提取所有数字
    numbers = re.findall(r'\b(\d{1,2})\b', text)

    # 过滤有效的1-49数字
    valid_numbers = [int(n) for n in numbers if 1 <= int(n) <= 49]

    if len(valid_numbers) < 7:
        return None

    # 前6个是主数字，最后1个是特别号
    main_numbers = valid_numbers[:6]
    special_number = valid_numbers[6]

    return {
        'period': period,
        'date': date_str,
        'numbers': main_numbers,
        'special': special_number,
        'full_text': text
    }


def convert_lottery_format(text: str) -> str:
    """
    转换彩票数据格式
    输入：包含生肖/五行信息的原始数据
    输出：简化格式
    """
    parsed = parse_lottery_data(text)
    if not parsed:
        return text

    numbers_str = ' '.join(map(str, parsed['numbers']))
    return f"第{parsed['period']}期 {parsed['date']} {numbers_str}+{parsed['special']}"


def format_numbers_display(numbers: List[int]) -> str:
    """格式化数字显示"""
    return ' '.join(str(n).zfill(2) for n in numbers)


def is_prime(n: int) -> bool:
    """判断是否为质数"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def get_number_properties(n: int) -> Dict[str, Any]:
    """获取数字属性"""
    return {
        'number': n,
        'is_prime': is_prime(n),
        'is_odd': n % 2 == 1,
        'is_even': n % 2 == 0,
        'is_small': n <= 24,  # 小数：1-24
        'is_large': n > 24,  # 大数：25-49
        'digit': n if n < 10 else n % 10,  # 尾数
        'zone': (n - 1) // 7 + 1,  # 区间：1-7
    }


# ============================================================================
# 数据管理类
# ============================================================================

class DataManager:
    """数据管理器 - 负责数据的持久化和加载"""

    def __init__(self, filepath: str = Config.DATA_FILE):
        self.filepath = filepath
        self.data = self.load_data()

    def load_data(self) -> Dict[str, Any]:
        """加载数据"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载数据失败: {e}")
        return self._get_default_data()

    def _get_default_data(self) -> Dict[str, Any]:
        """获取默认数据结构"""
        return {
            'history': [],  # 历史记录列表
            'predictions': [],  # 预测记录列表
            'settings': {
                'font_size': Config.DEFAULT_FONT_INDEX,
                'margin': Config.MARGIN_MEDIUM,
                'last_algorithm': 0
            }
        }

    def save_data(self) -> bool:
        """保存数据"""
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存数据失败: {e}")
            return False

    def add_history(self, record: Dict[str, Any]) -> None:
        """添加历史记录"""
        self.data['history'].insert(0, record)
        # 限制历史记录数量
        if len(self.data['history']) > 1000:
            self.data['history'] = self.data['history'][:1000]
        self.save_data()

    def get_history(self) -> List[Dict[str, Any]]:
        """获取历史记录"""
        return self.data['history']

    def update_history(self, index: int, record: Dict[str, Any]) -> bool:
        """更新历史记录"""
        if 0 <= index < len(self.data['history']):
            self.data['history'][index] = record
            self.save_data()
            return True
        return False

    def delete_history(self, indices: List[int]) -> None:
        """删除历史记录"""
        # 按索引降序排序，避免删除时索引变化
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(self.data['history']):
                self.data['history'].pop(idx)
        self.save_data()

    def clear_history(self) -> None:
        """清空历史记录"""
        self.data['history'] = []
        self.save_data()

    def get_settings(self) -> Dict[str, Any]:
        """获取设置"""
        return self.data['settings']

    def update_settings(self, key: str, value: Any) -> None:
        """更新设置"""
        self.data['settings'][key] = value
        self.save_data()


# ============================================================================
# 预测算法类
# ============================================================================

class PredictionAlgorithms:
    """
    预测算法集合
    包含12种不同的预测算法
    """

    def __init__(self, history: List[Dict[str, Any]]):
        self.history = history
        self.all_numbers = list(range(1, 50))

    def get_hot_cold_numbers(self) -> Tuple[List[int], List[int]]:
        """获取冷热号码"""
        if not self.history:
            return list(range(1, 50)), []

        # 统计所有出现过的数字
        all_nums = []
        for record in self.history:
            all_nums.extend(record.get('numbers', []))
            if 'special' in record:
                all_nums.append(record['special'])

        counter = Counter(all_nums)
        total = len(all_nums) if all_nums else 1

        # 计算频率
        freq = {n: counter.get(n, 0) / total for n in self.all_numbers}

        # 热号：出现频率高于平均的
        hot = sorted([n for n in self.all_numbers if freq[n] > 1.0 / 49],
                     key=lambda x: freq[x], reverse=True)
        # 冷号：出现频率低于平均的
        cold = sorted([n for n in self.all_numbers if freq[n] < 1.0 / 49],
                      key=lambda x: freq[x])

        return hot, cold

    def algorithm_comprehensive(self) -> List[int]:
        """
        算法1: 综合推荐
        综合多种因素进行推荐
        """
        scores = {n: 0 for n in self.all_numbers}

        if not self.history:
            # 无历史数据时随机
            return random.sample(self.all_numbers, 7)

        # 冷热分析
        hot, cold = self.get_hot_cold_numbers()
        for i, n in enumerate(hot[:15]):
            scores[n] += (15 - i) * 2
        for i, n in enumerate(cold[:15]):
            scores[n] += (15 - i) * 1.5

        # 遗漏值分析
        miss = self.get_miss_values()
        for n, m in miss.items():
            scores[n] += min(m, 20) * 0.5

        # 区间分布
        zones = self.get_zone_distribution()
        for zone, count in zones.items():
            if count < 2:  # 较少的区间
                zone_nums = [(zone - 1) * 7 + i for i in range(1, 8)]
                for n in zone_nums:
                    if 1 <= n <= 49:
                        scores[n] += 1

        # 奇偶平衡
        odd_count = sum(1 for n in scores if n % 2 == 1)
        even_count = len(scores) - odd_count
        if odd_count > even_count:
            for n in self.all_numbers:
                if n % 2 == 0:
                    scores[n] += 0.5
        else:
            for n in self.all_numbers:
                if n % 2 == 1:
                    scores[n] += 0.5

        # 大小平衡
        small = sum(1 for n in scores if n <= 24)
        large = len(scores) - small
        if small > large:
            for n in self.all_numbers:
                if n > 24:
                    scores[n] += 0.5
        else:
            for n in self.all_numbers:
                if n <= 24:
                    scores[n] += 0.5

        # 按分数排序并选择
        sorted_numbers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_numbers[:6]]

    def algorithm_hot_cold(self) -> List[int]:
        """
        算法2: 冷热数字算法
        根据数字出现频率进行预测
        """
        hot, cold = self.get_hot_cold_numbers()

        if not hot:
            return random.sample(self.all_numbers, 6)

        # 选择策略：60%热号 + 30%温号 + 10%冷号
        result = []
        result.extend(random.sample(hot[:20], min(4, len(hot))))

        # 中等频率的号码
        mid_freq = [n for n in self.all_numbers if n not in hot[:10] and n not in cold[:10]]
        result.extend(random.sample(mid_freq, min(1, len(mid_freq))))

        # 冷号补充
        while len(result) < 6 and cold:
            n = random.choice(cold[:10])
            if n not in result:
                result.append(n)

        return result[:6]

    def algorithm_odd_even(self) -> List[int]:
        """
        算法3: 单双算法
        根据奇偶分布进行预测
        """
        if not self.history:
            return random.sample(self.all_numbers, 6)

        # 统计近期奇偶分布
        recent = self.history[:10]
        odd_ratio = 0.5
        for record in recent:
            nums = record.get('numbers', [])
            odd_ratio = sum(1 for n in nums if n % 2 == 1) / max(len(nums), 1)

        # 目标是3:3或4:2的平衡
        target_odd = 3 if odd_ratio < 0.5 else 4
        target_even = 6 - target_odd

        result = []
        odd_nums = [n for n in self.all_numbers if n % 2 == 1]
        even_nums = [n for n in self.all_numbers if n % 2 == 0]

        result.extend(random.sample(odd_nums, min(target_odd, len(odd_nums))))
        result.extend(random.sample(even_nums, min(target_even, len(even_nums))))

        random.shuffle(result)
        return result[:6]

    def algorithm_size(self) -> List[int]:
        """
        算法4: 大小算法
        根据大小分布进行预测
        """
        if not self.history:
            return random.sample(self.all_numbers, 6)

        # 统计近期大小分布
        recent = self.history[:10]
        small_count = 0
        for record in recent:
            nums = record.get('numbers', [])
            small_count += sum(1 for n in nums if n <= 24)

        avg_small = small_count / max(len(recent), 1)

        # 目标是3:3或4:2的平衡
        target_small = 3 if avg_small < 3 else 4
        target_large = 6 - target_small

        result = []
        small_nums = [n for n in self.all_numbers if n <= 24]
        large_nums = [n for n in self.all_numbers if n > 24]

        result.extend(random.sample(small_nums, min(target_small, len(small_nums))))
        result.extend(random.sample(large_nums, min(target_large, len(large_nums))))

        random.shuffle(result)
        return result[:6]

    def get_miss_values(self) -> Dict[int, int]:
        """获取遗漏值"""
        miss = {n: 0 for n in self.all_numbers}

        if not self.history:
            return miss

        for record in self.history:
            nums = record.get('numbers', [])
            for n in nums:
                miss[n] += 1

        return miss

    def algorithm_miss(self) -> List[int]:
        """
        算法5: 遗漏值分析算法
        选择遗漏值较大的号码
        """
        miss = self.get_miss_values()

        # 按遗漏值排序
        sorted_by_miss = sorted(miss.items(), key=lambda x: x[1], reverse=True)

        # 选择遗漏值较大的，但保持多样性
        result = []
        for n, m in sorted_by_miss[:30]:
            if len(result) >= 6:
                break
            if m >= 1:  # 有遗漏的号码优先
                result.append(n)

        # 补充随机号码
        while len(result) < 6:
            n = random.choice(self.all_numbers)
            if n not in result:
                result.append(n)

        return result[:6]

    def algorithm_consecutive(self) -> List[int]:
        """
        算法6: 连号/邻号分析算法
        分析历史连号模式
        """
        if not self.history:
            return random.sample(self.all_numbers, 6)

        # 统计连号和邻号
        consecutive_count = 0
        adjacent_count = 0

        for record in self.history[:20]:
            nums = sorted(record.get('numbers', []))
            for i in range(len(nums) - 1):
                if nums[i + 1] - nums[i] == 1:
                    consecutive_count += 1
                elif nums[i + 1] - nums[i] == 2:
                    adjacent_count += 1

        result = []

        if consecutive_count > adjacent_count:
            # 避免连号
            while len(result) < 6:
                n = random.choice(self.all_numbers)
                if all(abs(n - r) > 1 for r in result):
                    result.append(n)
        else:
            # 可以包含一些邻号
            base = random.choice(self.all_numbers)
            result.append(base)
            for _ in range(3):
                n = base + random.choice([-2, -1, 1, 2])
                if 1 <= n <= 49 and n not in result:
                    result.append(n)

            while len(result) < 6:
                n = random.choice(self.all_numbers)
                if n not in result:
                    result.append(n)

        return result[:6]

    def algorithm_digit(self) -> List[int]:
        """
        算法7: 尾数分布算法
        分析尾数分布
        """
        if not self.history:
            return random.sample(self.all_numbers, 6)

        # 统计各尾数出现频率
        digit_counter = Counter()
        for record in self.history:
            nums = record.get('numbers', [])
            for n in nums:
                digit_counter[n % 10] += 1

        # 找出出现较少的尾数
        digit_freq = {d: digit_counter.get(d, 0) for d in range(10)}
        rare_digits = sorted(digit_freq, key=digit_freq.get)

        result = []
        for d in rare_digits[:4]:
            candidates = [n for n in self.all_numbers if n % 10 == d and n not in result]
            if candidates:
                result.append(random.choice(candidates))

        while len(result) < 6:
            n = random.choice(self.all_numbers)
            if n not in result:
                result.append(n)

        return result[:6]

    def get_zone_distribution(self) -> Dict[int, int]:
        """获取区间分布"""
        zones = {i: 0 for i in range(1, 8)}
        for record in self.history:
            nums = record.get('numbers', [])
            for n in nums:
                zone = (n - 1) // 7 + 1
                zones[zone] += 1
        return zones

    def algorithm_zone(self) -> List[int]:
        """
        算法8: 区间分布算法
        分析区间分布
        """
        zones = self.get_zone_distribution()

        # 找出出现较少的区间
        sorted_zones = sorted(zones.items(), key=lambda x: x[1])
        result = []

        for zone, _ in sorted_zones[:4]:
            nums_in_zone = [(zone - 1) * 7 + i for i in range(1, 8)]
            nums_in_zone = [n for n in nums_in_zone if 1 <= n <= 49]
            if nums_in_zone:
                result.append(random.choice(nums_in_zone))

        while len(result) < 6:
            n = random.choice(self.all_numbers)
            if n not in result:
                result.append(n)

        return result[:6]

    def algorithm_roulette(self) -> List[int]:
        """
        算法9: 轮盘赌选择算法
        模拟轮盘赌选择
        """
        # 为每个号码设置权重
        weights = {n: 1.0 for n in self.all_numbers}

        if self.history:
            # 近期出现过的号码权重降低
            recent = set()
            for record in self.history[:5]:
                recent.update(record.get('numbers', []))

            for n in recent:
                weights[n] *= 0.7

            # 遗漏值增加权重
            miss = self.get_miss_values()
            for n, m in miss.items():
                weights[n] *= (1 + min(m, 10) * 0.1)

        # 轮盘赌选择
        numbers = list(weights.keys())
        weight_list = list(weights.values())
        total_weight = sum(weight_list)
        probs = [w / total_weight for w in weight_list]

        result = []
        for _ in range(6):
            n = random.choices(numbers, weights=probs, k=1)[0]
            if n not in result:
                result.append(n)
            else:
                # 重新选择
                for nn in random.sample(numbers, len(numbers)):
                    if nn not in result:
                        result.append(nn)
                        break

        return result[:6]

    def algorithm_similarity(self) -> List[int]:
        """
        算法10: 历史相似性算法
        找出与近期数据最相似的历史记录
        """
        if len(self.history) < 5:
            return random.sample(self.all_numbers, 6)

        # 获取最近一期作为参考
        reference = self.history[0].get('numbers', [])
        if not reference:
            return random.sample(self.all_numbers, 6)

        # 找相似记录
        similar_records = []
        for record in self.history[1:]:
            nums = record.get('numbers', [])
            # 计算相似度
            common = len(set(reference) & set(nums))
            if common >= 2:
                similar_records.append((record, common))

        if not similar_records:
            return random.sample(self.all_numbers, 6)

        # 按相似度排序
        similar_records.sort(key=lambda x: x[1], reverse=True)

        # 选择相似记录的下一位
        result = []
        for record, _ in similar_records[:3]:
            next_nums = record.get('numbers', [])
            for n in random.sample(next_nums, min(2, len(next_nums))):
                if n not in result:
                    result.append(n)

        while len(result) < 6:
            n = random.choice(self.all_numbers)
            if n not in result:
                result.append(n)

        return result[:6]

    def algorithm_poisson(self) -> List[int]:
        """
        算法11: 泊松概率分布算法
        使用泊松分布进行预测
        """
        if not self.history:
            return random.sample(self.all_numbers, 6)

        # 计算每个数字出现次数的均值
        all_nums = []
        for record in self.history:
            all_nums.extend(record.get('numbers', []))

        if not all_nums:
            return random.sample(self.all_numbers, 6)

        counter = Counter(all_nums)
        mean = sum(counter.values()) / len(counter) if counter else 1

        # 使用泊松分布计算概率
        result = []
        for n in self.all_numbers:
            count = counter.get(n, 0)
            # 泊松概率
            poisson_prob = (mean ** count * math.exp(-mean)) / math.factorial(count) if count < 10 else 0

            # 遗漏增加概率
            miss = self.get_miss_values().get(n, 0)
            adjusted_prob = poisson_prob * (1 + miss * 0.1)

            result.append((n, adjusted_prob))

        # 按概率排序
        result.sort(key=lambda x: x[1], reverse=True)

        # 选择概率较高的，但保持多样性
        selected = [n for n, _ in result[:20]]
        return random.sample(selected, min(6, len(selected)))

    def algorithm_mystical(self) -> List[int]:
        """
        算法12: 玄学算法
        结合多种玄学因素
        """
        # 生肖、五行等（简化模拟）
        zodiac_years = {
            0: '猴', 1: '鸡', 2: '狗', 3: '猪',
            4: '鼠', 5: '牛', 6: '虎', 7: '兔',
            8: '龙', 9: '蛇', 10: '马', 11: '羊'
        }

        wuxing = ['木', '金', '土', '水', '火']

        # 当前时间相关因子
        now = datetime.now()
        year_zodiac = zodiac_years.get(now.year % 12, '鼠')
        month_wuxing = wuxing[(now.month - 1) % 5]

        # 生成种子
        seed_str = f"{year_zodiac}{month_wuxing}{now.day}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)

        random.seed(seed)

        # 综合选择
        result = []

        # 金木水火土各选一个
        for i in range(5):
            base = (i + 1) * 7 + (now.day % 7)
            n = ((base + seed) % 49) + 1
            result.append(n)

        # 最后一个随机
        result.append(random.randint(1, 49))

        # 去重并补充
        result = list(set(result))
        while len(result) < 6:
            n = random.randint(1, 49)
            if n not in result:
                result.append(n)

        random.seed()  # 重置随机种子

        return result[:6]

    def predict(self, algorithm_index: int) -> Tuple[List[int], int]:
        """
        执行预测

        Args:
            algorithm_index: 算法索引 (0-11)

        Returns:
            (主号码列表, 特别号)
        """
        algorithms = [
            self.algorithm_comprehensive,
            self.algorithm_hot_cold,
            self.algorithm_odd_even,
            self.algorithm_size,
            self.algorithm_miss,
            self.algorithm_consecutive,
            self.algorithm_digit,
            self.algorithm_zone,
            self.algorithm_roulette,
            self.algorithm_similarity,
            self.algorithm_poisson,
            self.algorithm_mystical,
        ]

        if algorithm_index < 0 or algorithm_index >= len(algorithms):
            algorithm_index = 0

        main_numbers = algorithms[algorithm_index]()

        # 生成特别号（使用不同策略）
        used_numbers = set(main_numbers)
        special = random.choice([n for n in self.all_numbers if n not in used_numbers])

        return main_numbers, special


# ============================================================================
# 预测模型类 (使用torch)
# ============================================================================

class PredictionModel:
    """
    预测模型 - 基于神经网络的预测
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        if HAS_TORCH:
            self._build_model()

    def _build_model(self):
        """构建神经网络模型"""

        class LotteryNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(54, 128)  # 输入：49个数字出现次数 + 5个特征
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 32)
                self.fc4 = nn.Linear(32, 7)  # 输出：7个预测号码

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = torch.relu(self.fc3(x))
                x = self.fc4(x)
                return x

        self.model = LotteryNet()

    def prepare_features(self, history: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """准备特征"""
        if not history or not HAS_NUMPY:
            return None

        features = []

        for record in history[:50]:  # 使用最近50条记录
            nums = record.get('numbers', [])
            special = record.get('special', 0)

            # 49个数字的出现次数
            freq = [0] * 49
            for n in nums:
                if 1 <= n <= 49:
                    freq[n - 1] = 1
            if 1 <= special <= 49:
                freq[special - 1] = 1

            # 附加特征
            props = {
                'odd_count': sum(1 for n in nums if n % 2 == 1),
                'even_count': sum(1 for n in nums if n % 2 == 0),
                'small_count': sum(1 for n in nums if n <= 24),
                'large_count': sum(1 for n in nums if n > 24),
                'avg': sum(nums) / len(nums) if nums else 0,
            }

            features.append(freq + [props['odd_count'], props['even_count'],
                                    props['small_count'], props['large_count'], props['avg']])

        if not features:
            return None

        # 填充到相同长度
        while len(features) < 50:
            features.append([0] * 54)

        return np.array(features[:50])

    def predict_nn(self, history: List[Dict[str, Any]]) -> Optional[Tuple[List[int], int]]:
        """使用神经网络预测"""
        if not HAS_TORCH or not self.model:
            return None

        features = self.prepare_features(history)
        if features is None:
            return None

        try:
            self.model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(features.mean(axis=0).reshape(1, -1))
                output = self.model(input_tensor)

                # 将输出转换为号码
                probs = torch.softmax(output, dim=1).numpy()[0]
                numbers = np.argsort(probs)[-7:][::-1] + 1  # 取概率最高的7个

            main_numbers = numbers[:6].tolist()
            special = numbers[6]

            return main_numbers, int(special)
        except Exception as e:
            print(f"神经网络预测失败: {e}")
            return None


# ============================================================================
# 自定义样式类
# ============================================================================

class StyleSheet:
    """样式表生成器"""

    @staticmethod
    def get_stylesheet(
            bg_color: str = Config.COLORS['white'],
            primary: str = Config.COLORS['primary'],
            font_size: int = 12
    ) -> str:
        """生成完整的样式表"""
        return f"""
        /* 全局样式 */
        QMainWindow, QWidget {{
            background-color: {bg_color};
            color: {Config.COLORS['text']};
            font-size: {font_size}px;
        }}

        /* 标签页样式 */
        QTabWidget::pane {{
            border: 1px solid {Config.COLORS['border']};
            border-radius: 4px;
            background-color: {bg_color};
        }}

        QTabBar::tab {{
            background-color: {Config.COLORS['white']};
            color: {Config.COLORS['text_secondary']};
            padding: 8px 16px;
            margin-right: 2px;
            border: 1px solid {Config.COLORS['border']};
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}

        QTabBar::tab:selected {{
            background-color: {primary};
            color: {Config.COLORS['white']};
            font-weight: bold;
        }}

        QTabBar::tab:hover {{
            background-color: {Config.COLORS['primary_light']};
        }}

        /* 按钮样式 */
        QPushButton {{
            background-color: {primary};
            color: {Config.COLORS['white']};
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            min-width: 80px;
            font-size: {font_size}px;
        }}

        QPushButton:hover {{
            background-color: {Config.COLORS['primary_dark']};
        }}

        QPushButton:pressed {{
            background-color: {Config.COLORS['primary_dark']};
        }}

        QPushButton:disabled {{
            background-color: {Config.COLORS['border']};
            color: {Config.COLORS['text_secondary']};
        }}

        /* 成功按钮 */
        QPushButton[class="success"] {{
            background-color: {Config.COLORS['success']};
        }}

        QPushButton[class="success"]:hover {{
            background-color: #388E3C;
        }}

        /* 危险按钮 */
        QPushButton[class="danger"] {{
            background-color: {Config.COLORS['danger']};
        }}

        QPushButton[class="danger"]:hover {{
            background-color: #D32F2F;
        }}

        /* 警告按钮 */
        QPushButton[class="warning"] {{
            background-color: {Config.COLORS['warning']};
        }}

        /* 输入框样式 */
        QLineEdit, QTextEdit {{
            background-color: {bg_color};
            color: {Config.COLORS['text']};
            border: 1px solid {Config.COLORS['border']};
            border-radius: 4px;
            padding: 6px;
            font-size: {font_size}px;
            selection-background-color: {Config.COLORS['primary_light']};
        }}

        QLineEdit:focus, QTextEdit:focus {{
            border: 2px solid {primary};
        }}

        /* 组合框样式 */
        QComboBox {{
            background-color: {bg_color};
            color: {Config.COLORS['text']};
            border: 1px solid {Config.COLORS['border']};
            border-radius: 4px;
            padding: 6px;
            font-size: {font_size}px;
        }}

        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}

        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid {Config.COLORS['text_secondary']};
        }}

        QComboBox QAbstractItemView {{
            background-color: {bg_color};
            color: {Config.COLORS['text']};
            border: 1px solid {Config.COLORS['border']};
            selection-background-color: {Config.COLORS['primary_light']};
        }}

        /* 表格样式 */
        QTableWidget {{
            background-color: {bg_color};
            color: {Config.COLORS['text']};
            border: 1px solid {Config.COLORS['border']};
            gridline-color: {Config.COLORS['border']};
            font-size: {font_size}px;
        }}

        QTableWidget::item {{
            padding: 4px;
        }}

        QTableWidget::item:selected {{
            background-color: {Config.COLORS['primary_light']};
            color: {Config.COLORS['text']};
        }}

        QHeaderView::section {{
            background-color: {primary};
            color: {Config.COLORS['white']};
            padding: 6px;
            border: none;
            font-weight: bold;
        }}

        /* 滚动条样式 */
        QScrollBar:vertical {{
            background-color: {bg_color};
            width: 12px;
            border: none;
        }}

        QScrollBar::handle:vertical {{
            background-color: {Config.COLORS['border']};
            border-radius: 6px;
            min-height: 20px;
        }}

        QScrollBar::handle:vertical:hover {{
            background-color: {Config.COLORS['text_secondary']};
        }}

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}

        QScrollBar:horizontal {{
            background-color: {bg_color};
            height: 12px;
            border: none;
        }}

        QScrollBar::handle:horizontal {{
            background-color: {Config.COLORS['border']};
            border-radius: 6px;
            min-width: 20px;
        }}

        QScrollBar::handle:horizontal:hover {{
            background-color: {Config.COLORS['text_secondary']};
        }}

        /* 分隔条样式 */
        QSplitter::handle {{
            background-color: {primary};
        }}

        QSplitter::handle:horizontal {{
            width: 4px;
        }}

        QSplitter::handle:vertical {{
            height: 4px;
        }}

        /* 工具栏样式 */
        QToolBar {{
            background-color: {bg_color};
            border: none;
            spacing: 4px;
            padding: 4px;
        }}

        QToolBar::separator {{
            background-color: {Config.COLORS['border']};
            width: 1px;
            margin: 4px;
        }}

        /* 状态栏样式 */
        QStatusBar {{
            background-color: {bg_color};
            color: {Config.COLORS['text_secondary']};
            border-top: 1px solid {Config.COLORS['border']};
        }}

        /* 列表样式 */
        QListWidget {{
            background-color: {bg_color};
            color: {Config.COLORS['text']};
            border: 1px solid {Config.COLORS['border']};
            border-radius: 4px;
            font-size: {font_size}px;
        }}

        QListWidget::item {{
            padding: 6px;
            border-bottom: 1px solid {Config.COLORS['border']};
        }}

        QListWidget::item:selected {{
            background-color: {Config.COLORS['primary_light']};
        }}

        /* 分组框样式 */
        QGroupBox {{
            border: 1px solid {Config.COLORS['border']};
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
        }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: {primary};
        }}

        /* 进度条样式 */
        QProgressBar {{
            background-color: {Config.COLORS['border']};
            border: none;
            border-radius: 4px;
            text-align: center;
            height: 20px;
        }}

        QProgressBar::chunk {{
            background-color: {primary};
            border-radius: 4px;
        }}

        /* 复选框样式 */
        QCheckBox {{
            color: {Config.COLORS['text']};
            font-size: {font_size}px;
        }}

        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 2px solid {primary};
            border-radius: 3px;
            background-color: {bg_color};
        }}

        QCheckBox::indicator:checked {{
            background-color: {primary};
        }}

        /* 标签样式 */
        QLabel {{
            color: {Config.COLORS['text']};
            font-size: {font_size}px;
        }}

        /* 旋转框样式 */
        QSpinBox, QDoubleSpinBox {{
            background-color: {bg_color};
            color: {Config.COLORS['text']};
            border: 1px solid {Config.COLORS['border']};
            border-radius: 4px;
            padding: 4px;
            font-size: {font_size}px;
        }}

        /* 滑块样式 */
        QSlider::groove:horizontal {{
            border: 1px solid {Config.COLORS['border']};
            height: 6px;
            background-color: {bg_color};
            border-radius: 3px;
        }}

        QSlider::handle:horizontal {{
            background-color: {primary};
            border: none;
            width: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }}

        QSlider::sub-page:horizontal {{
            background-color: {primary};
            border-radius: 3px;
        }}
        """


# ============================================================================
# 工具栏按钮类
# ============================================================================

class ToolbarButton(QPushButton):
    """工具栏按钮"""

    def __init__(
            self,
            text: str,
            color: str = Config.COLORS['primary'],
            icon: Optional[QIcon] = None,
            parent: Optional[QWidget] = None
    ):
        super().__init__(text, parent)
        self.setFixedHeight(32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        if icon:
            self.setIcon(icon)

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: {Config.COLORS['white']};
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {Config.COLORS['primary_dark']};
            }}
        """)


# ============================================================================
# 标签页1: 数据转换面板
# ============================================================================

class DataConversionPanel(QWidget):
    """
    数据转换面板
    用于粘贴和转换各种格式的数据
    """

    def __init__(self, main_window: 'MainWindow', parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 标题
        title_label = QLabel("📋 数据格式转换")
        title_label.setStyleSheet(f"""
            QLabel {{
                font-size: 16px;
                font-weight: bold;
                color: {Config.COLORS['primary']};
            }}
        """)
        layout.addWidget(title_label)

        # 主分隔条 - 输入区域与输出区域
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {Config.COLORS['border']};
                height: 6px;
                margin: 2px 0px;
                border-radius: 3px;
            }}
            QSplitter::handle:hover {{
                background-color: {Config.COLORS['primary']};
            }}
        """)

        # 输入区域 - 添加滚动条
        input_widget = QWidget()
        input_outer_layout = QVBoxLayout(input_widget)
        input_outer_layout.setContentsMargins(0, 0, 0, 0)

        input_scroll = QScrollArea()
        input_scroll.setWidgetResizable(True)
        input_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {Config.COLORS['border']};
                border-radius: 6px;
                background-color: white;
            }}
            QScrollBar:vertical {{
                background-color: {Config.COLORS['white']};
                width: 12px;
                border-radius: 6px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {Config.COLORS['primary']};
                min-height: 30px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {Config.COLORS['primary_dark']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background-color: {Config.COLORS['white']};
            }}
            QScrollBar:horizontal {{
                background-color: {Config.COLORS['white']};
                height: 12px;
                border-radius: 6px;
                margin: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {Config.COLORS['primary']};
                min-width: 30px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {Config.COLORS['primary_dark']};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
                background-color: {Config.COLORS['white']};
            }}
        """)

        input_group = QGroupBox("📥 粘贴原始数据")
        input_layout = QVBoxLayout(input_group)

        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText(
            "请粘贴原始数据，例如：\n"
            "第116期最新开奖结果 2026年04月26日 15 龙/水 46 鸡/木 16 兔/木 10 鸡/火 48 羊/火 33 狗/火 22 鸡/水\n\n"
            "支持的格式：\n"
            "• 包含生肖/五行信息的数据\n"
            "• 纯数字格式（如：116 2026-04-26 15 46 16 10 48 33 22）\n"
            "• 其他包含数字的文本"
        )
        self.input_text.setMinimumHeight(150)
        input_layout.addWidget(self.input_text)

        # 按钮行
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        convert_btn = ToolbarButton("🔄 转换", Config.COLORS['primary'])
        convert_btn.clicked.connect(self.convert_data)
        button_layout.addWidget(convert_btn)

        clear_btn = ToolbarButton("🗑️ 清空", Config.COLORS['danger'])
        clear_btn.clicked.connect(self.clear_input)
        button_layout.addWidget(clear_btn)

        input_layout.addLayout(button_layout)

        input_scroll.setWidget(input_group)
        input_outer_layout.addWidget(input_scroll)
        main_splitter.addWidget(input_widget)

        # 输出区域 - 添加滚动条
        output_widget = QWidget()
        output_outer_layout = QVBoxLayout(output_widget)
        output_outer_layout.setContentsMargins(0, 0, 0, 0)

        output_scroll = QScrollArea()
        output_scroll.setWidgetResizable(True)
        output_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {Config.COLORS['border']};
                border-radius: 6px;
                background-color: white;
            }}
            QScrollBar:vertical {{
                background-color: {Config.COLORS['white']};
                width: 12px;
                border-radius: 6px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {Config.COLORS['success']};
                min-height: 30px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {Config.COLORS['primary_dark']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background-color: {Config.COLORS['white']};
            }}
            QScrollBar:horizontal {{
                background-color: {Config.COLORS['white']};
                height: 12px;
                border-radius: 6px;
                margin: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {Config.COLORS['success']};
                min-width: 30px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {Config.COLORS['primary_dark']};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
                background-color: {Config.COLORS['white']};
            }}
        """)

        output_group = QGroupBox("📤 转换结果预览")
        output_layout = QVBoxLayout(output_group)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(100)
        self.output_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Config.COLORS['success_light']};
                color: {Config.COLORS['text']};
                border: 1px solid {Config.COLORS['success']};
                border-radius: 4px;
            }}
        """)
        output_layout.addWidget(self.output_text)

        # 保存按钮
        save_layout = QHBoxLayout()
        save_layout.addStretch()

        save_btn = ToolbarButton("💾 保存到分析面板", Config.COLORS['success'])
        save_btn.clicked.connect(self.save_to_analysis)
        save_layout.addWidget(save_btn)

        output_layout.addLayout(save_layout)

        output_scroll.setWidget(output_group)
        output_outer_layout.addWidget(output_scroll)
        main_splitter.addWidget(output_widget)

        # 使用说明 - 添加滚动条
        help_widget = QWidget()
        help_outer_layout = QVBoxLayout(help_widget)
        help_outer_layout.setContentsMargins(0, 0, 0, 0)

        help_scroll = QScrollArea()
        help_scroll.setWidgetResizable(True)
        help_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {Config.COLORS['info']};
                border-radius: 6px;
                background-color: white;
            }}
            QScrollBar:vertical {{
                background-color: {Config.COLORS['white']};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {Config.COLORS['info']};
                min-height: 20px;
                border-radius: 6px;
            }}
        """)

        help_label = QLabel(
            "💡 使用说明：\n"
            "1. 在上方粘贴需要转换的数据\n"
            "2. 点击「转换」按钮进行格式转换\n"
            "3. 预览转换结果\n"
            "4. 点击「保存到分析面板」将数据添加到分析系统"
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet(f"""
            QLabel {{
                background-color: {Config.COLORS['info']}22;
                color: {Config.COLORS['text']};
                border-radius: 4px;
                padding: 10px;
                font-size: 12px;
            }}
        """)
        help_scroll.setWidget(help_label)
        help_outer_layout.addWidget(help_scroll)
        main_splitter.addWidget(help_widget)

        main_splitter.setSizes([200, 200, 100])  # 设置初始比例
        layout.addWidget(main_splitter, 1)

    def convert_data(self):
        """转换数据"""
        text = self.input_text.toPlainText().strip()

        if not text:
            self.output_text.setHtml(
                f'<span style="color: {Config.COLORS["danger"]};">⚠️ 请输入要转换的数据</span>'
            )
            return

        # 尝试转换
        converted = convert_lottery_format(text)

        # 解析数据
        parsed = parse_lottery_data(text)

        if parsed:
            # 显示详细结果
            result_html = f'''
            <div style="color: {Config.COLORS["success"]}; font-weight: bold;">
                ✅ 转换成功！
            </div>
            <br/>
            <b>期号：</b>第{parsed['period']}期<br/>
            <b>日期：</b>{parsed['date']}<br/>
            <b>主号码：</b><span style="color: {Config.COLORS["primary"]}; font-size: 16px; font-weight: bold;">
                {' '.join(str(n).zfill(2) for n in parsed['numbers'])}
            </span><br/>
            <b>特别号：</b><span style="color: {Config.COLORS["danger"]}; font-size: 16px; font-weight: bold;">
                {str(parsed['special']).zfill(2)}
            </span><br/>
            <br/>
            <b>简化格式：</b><br/>
            <div style="background-color: white; padding: 8px; border-radius: 4px; margin-top: 5px;">
                {converted}
            </div>
            '''
            self.output_text.setHtml(result_html)
            self.last_converted = parsed
        else:
            # 尝试提取纯数字
            numbers = re.findall(r'\b(\d{1,2})\b', text)
            valid_nums = [int(n) for n in numbers if 1 <= int(n) <= 49]

            if len(valid_nums) >= 7:
                period_match = re.search(r'第\s*(\d+)', text)
                period = period_match.group(1) if period_match else "未知"
                date_match = re.search(r'(\d{4})[年\-\.](\d{1,2})[月\-\.](\d{1,2})', text)
                if date_match:
                    date = f"{date_match.group(1)}-{date_match.group(2).zfill(2)}-{date_match.group(3).zfill(2)}"
                else:
                    date = datetime.now().strftime('%Y-%m-%d')

                self.last_converted = {
                    'period': period,
                    'date': date,
                    'numbers': valid_nums[:6],
                    'special': valid_nums[6]
                }

                result_html = f'''
                <div style="color: {Config.COLORS["success"]}; font-weight: bold;">
                    ✅ 提取成功！
                </div>
                <br/>
                <b>期号：</b>第{period}<br/>
                <b>日期：</b>{date}<br/>
                <b>主号码：</b><span style="color: {Config.COLORS["primary"]}; font-size: 16px; font-weight: bold;">
                    {' '.join(str(n).zfill(2) for n in valid_nums[:6])}
                </span><br/>
                <b>特别号：</b><span style="color: {Config.COLORS["danger"]}; font-size: 16px; font-weight: bold;">
                    {str(valid_nums[6]).zfill(2)}
                </span>
                '''
                self.output_text.setHtml(result_html)
            else:
                self.output_text.setHtml(
                    f'<span style="color: {Config.COLORS["danger"]};">'
                    f'⚠️ 无法识别的数据格式。请确保数据包含有效的数字（1-49）。'
                    f'</span>'
                )
                self.last_converted = None

    def clear_input(self):
        """清空输入"""
        self.input_text.clear()
        self.output_text.clear()
        self.last_converted = None

    def save_to_analysis(self):
        """保存到分析面板"""
        if not hasattr(self, 'last_converted') or not self.last_converted:
            self.output_text.setHtml(
                f'<span style="color: {Config.COLORS["danger"]};">'
                f'⚠️ 请先转换数据'
                f'</span>'
            )
            return

        # 添加到分析面板
        self.main_window.add_history_record(self.last_converted)

        # 切换到分析面板
        self.main_window.tab_widget.setCurrentIndex(1)

        # 显示成功消息
        self.output_text.setHtml(
            f'<span style="color: {Config.COLORS["success"]}; font-weight: bold;">'
            f'✅ 数据已保存到分析面板！'
            f'</span>'
        )


# ============================================================================
# 数字按钮类
# ============================================================================

class NumberButton(QPushButton):
    """数字按钮"""

    def __init__(self, number: int, parent: Optional[QWidget] = None):
        super().__init__(str(number), parent)
        self.number = number
        self.is_selected = False
        self.setFixedSize(50, 50)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update_style()

    def update_style(self):
        """更新样式"""
        if self.is_selected:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Config.COLORS['primary']};
                    color: {Config.COLORS['white']};
                    border: 2px solid {Config.COLORS['primary_dark']};
                    border-radius: 25px;
                    font-size: 16px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {Config.COLORS['primary_dark']};
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Config.COLORS['white']};
                    color: {Config.COLORS['text']};
                    border: 2px solid {Config.COLORS['border']};
                    border-radius: 25px;
                    font-size: 16px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {Config.COLORS['primary_light']};
                    border-color: {Config.COLORS['primary']};
                }}
            """)

    def set_selected(self, selected: bool):
        """设置选中状态"""
        self.is_selected = selected
        self.update_style()


# ============================================================================
# 历史记录表格类
# ============================================================================

class HistoryTable(QTableWidget):
    """历史记录表格"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(['期号', '日期', '主号码', '特别号', '操作'])
        self.horizontalHeader().setStretchLastSection(True)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setMinimumHeight(200)


# ============================================================================
# 标签页2: 分析预测面板
# ============================================================================

class AnalysisPanel(QWidget):
    """
    分析预测面板
    包含49个数字随机抽取系统和多种预测算法
    """

    def __init__(self, main_window: 'MainWindow', parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main_window = main_window
        self.data_manager = main_window.data_manager
        self.current_prediction = None
        self.algorithm = None
        self.init_ui()
        self.load_history()

    def init_ui(self):
        """初始化UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 创建水平分割器
        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧面板
        left_widget = self.create_left_panel()
        h_splitter.addWidget(left_widget)

        # 右侧面板
        right_widget = self.create_right_panel()
        h_splitter.addWidget(right_widget)

        # 设置初始比例
        h_splitter.setStretchFactor(0, 1)
        h_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(h_splitter, 1)

    def create_left_panel(self) -> QWidget:
        """创建左侧面板"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # 标题
        title_label = QLabel("🎲 数字预测系统")
        title_label.setStyleSheet(f"""
            QLabel {{
                font-size: 16px;
                font-weight: bold;
                color: {Config.COLORS['primary']};
            }}
        """)
        layout.addWidget(title_label)

        # 左侧主分隔条 - 算法选择区 / 49个数字区 / 预测结果区
        left_main_splitter = QSplitter(Qt.Orientation.Vertical)
        left_main_splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {Config.COLORS['border']};
                height: 6px;
                margin: 2px 0px;
                border-radius: 3px;
            }}
            QSplitter::handle:hover {{
                background-color: {Config.COLORS['primary']};
            }}
        """)

        # 算法选择区 - 添加滚动条
        algo_widget = QWidget()
        algo_outer_layout = QVBoxLayout(algo_widget)
        algo_outer_layout.setContentsMargins(0, 0, 0, 0)

        algo_scroll = QScrollArea()
        algo_scroll.setWidgetResizable(True)
        algo_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {Config.COLORS['border']};
                border-radius: 6px;
                background-color: white;
            }}
            QScrollBar:vertical {{
                background-color: {Config.COLORS['white']};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {Config.COLORS['purple']};
                min-height: 20px;
                border-radius: 6px;
            }}
        """)

        algo_group = QGroupBox("⚙️ 选择预测算法")
        algo_layout = QVBoxLayout(algo_group)

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(Config.ALGORITHMS)
        self.algorithm_combo.currentIndexChanged.connect(self.on_algorithm_changed)
        algo_layout.addWidget(self.algorithm_combo)

        # 预测按钮
        btn_layout = QHBoxLayout()
        predict_btn = ToolbarButton("🔮 预测", Config.COLORS['primary'])
        predict_btn.clicked.connect(self.predict)
        btn_layout.addWidget(predict_btn)

        extract_btn = ToolbarButton("🎯 抽取", Config.COLORS['success'])
        extract_btn.clicked.connect(self.extract)
        btn_layout.addWidget(extract_btn)

        algo_layout.addLayout(btn_layout)

        algo_scroll.setWidget(algo_group)
        algo_outer_layout.addWidget(algo_scroll)
        left_main_splitter.addWidget(algo_widget)

        # 49个数字区域 - 添加滚动条
        numbers_widget = QWidget()
        numbers_outer_layout = QVBoxLayout(numbers_widget)
        numbers_outer_layout.setContentsMargins(0, 0, 0, 0)

        numbers_scroll = QScrollArea()
        numbers_scroll.setWidgetResizable(True)
        numbers_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {Config.COLORS['border']};
                border-radius: 6px;
                background-color: white;
            }}
            QScrollBar:vertical {{
                background-color: {Config.COLORS['white']};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {Config.COLORS['warning']};
                min-height: 20px;
                border-radius: 6px;
            }}
            QScrollBar:horizontal {{
                background-color: {Config.COLORS['white']};
                height: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {Config.COLORS['warning']};
                min-width: 20px;
                border-radius: 6px;
            }}
        """)

        numbers_group = QGroupBox("🎯 49个数字 (1-49)")
        numbers_layout = QVBoxLayout(numbers_group)

        self.number_buttons = {}
        for row in range(7):
            row_layout = QHBoxLayout()
            row_layout.setSpacing(5)
            for col in range(7):
                num = row * 7 + col + 1
                btn = NumberButton(num)
                btn.clicked.connect(lambda checked, n=num: self.on_number_clicked(n))
                self.number_buttons[num] = btn
                row_layout.addWidget(btn)
            row_layout.addStretch()
            numbers_layout.addLayout(row_layout)

        numbers_scroll.setWidget(numbers_group)
        numbers_outer_layout.addWidget(numbers_scroll)
        left_main_splitter.addWidget(numbers_widget)

        # 预测结果与第七位预判分隔条
        predict_splitter = QSplitter(Qt.Orientation.Vertical)
        predict_splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {Config.COLORS['border']};
                height: 6px;
                margin: 2px 0px;
                border-radius: 3px;
            }}
            QSplitter::handle:hover {{
                background-color: {Config.COLORS['primary']};
            }}
        """)

        # 预测结果展示 - 添加滚动条
        result_widget = QWidget()
        result_outer_layout = QVBoxLayout(result_widget)
        result_outer_layout.setContentsMargins(0, 0, 0, 0)

        result_scroll = QScrollArea()
        result_scroll.setWidgetResizable(True)
        result_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {Config.COLORS['border']};
                border-radius: 6px;
                background-color: white;
            }}
            QScrollBar:vertical {{
                background-color: {Config.COLORS['white']};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {Config.COLORS['primary']};
                min-height: 20px;
                border-radius: 6px;
            }}
        """)

        result_group = QGroupBox("📊 预测结果")
        result_layout = QVBoxLayout(result_group)

        self.result_label = QLabel("点击「预测」或「抽取」按钮获取结果")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet(f"""
            QLabel {{
                background-color: {Config.COLORS['primary_light']};
                color: {Config.COLORS['text']};
                border-radius: 8px;
                padding: 15px;
                font-size: 14px;
            }}
        """)
        result_layout.addWidget(self.result_label)

        result_scroll.setWidget(result_group)
        result_outer_layout.addWidget(result_scroll)
        predict_splitter.addWidget(result_widget)

        # 预判功能 - 添加滚动条
        forecast_widget = QWidget()
        forecast_outer_layout = QVBoxLayout(forecast_widget)
        forecast_outer_layout.setContentsMargins(0, 0, 0, 0)

        forecast_scroll = QScrollArea()
        forecast_scroll.setWidgetResizable(True)
        forecast_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {Config.COLORS['border']};
                border-radius: 6px;
                background-color: white;
            }}
            QScrollBar:vertical {{
                background-color: {Config.COLORS['white']};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {Config.COLORS['success']};
                min-height: 20px;
                border-radius: 6px;
            }}
        """)

        forecast_group = QGroupBox("🔮 第七位预判")
        forecast_layout = QVBoxLayout(forecast_group)

        forecast_row1 = QHBoxLayout()
        forecast_row1.addWidget(QLabel("大小:"))
        self.size_forecast = QLabel("--")
        forecast_row1.addWidget(self.size_forecast)
        forecast_row1.addStretch()

        self.size_big_btn = ToolbarButton("大", Config.COLORS['primary'])
        self.size_big_btn.setFixedWidth(40)
        self.size_big_btn.clicked.connect(lambda: self.set_forecast_size("大"))
        forecast_row1.addWidget(self.size_big_btn)

        self.size_small_btn = ToolbarButton("小", Config.COLORS['info'])
        self.size_small_btn.setFixedWidth(40)
        self.size_small_btn.clicked.connect(lambda: self.set_forecast_size("小"))
        forecast_row1.addWidget(self.size_small_btn)

        forecast_layout.addLayout(forecast_row1)

        forecast_row2 = QHBoxLayout()
        forecast_row2.addWidget(QLabel("单双:"))
        self.odd_forecast = QLabel("--")
        forecast_row2.addWidget(self.odd_forecast)
        forecast_row2.addStretch()

        self.odd_btn = ToolbarButton("单", Config.COLORS['purple'])
        self.odd_btn.setFixedWidth(40)
        self.odd_btn.clicked.connect(lambda: self.set_forecast_odd("单"))
        forecast_row2.addWidget(self.odd_btn)

        self.even_btn = ToolbarButton("双", Config.COLORS['pink'])
        self.even_btn.setFixedWidth(40)
        self.even_btn.clicked.connect(lambda: self.set_forecast_odd("双"))
        forecast_row2.addWidget(self.even_btn)

        forecast_layout.addLayout(forecast_row2)

        forecast_row3 = QHBoxLayout()
        forecast_row3.addWidget(QLabel("尾数:"))
        self.digit_forecast = QLabel("--")
        forecast_row3.addWidget(self.digit_forecast)
        forecast_row3.addStretch()

        self.digit_big_btn = ToolbarButton("大(5-9)", Config.COLORS['warning'])
        self.digit_big_btn.clicked.connect(lambda: self.set_forecast_digit("大"))
        forecast_row3.addWidget(self.digit_big_btn)

        self.digit_small_btn = ToolbarButton("小(0-4)", Config.COLORS['info'])
        self.digit_small_btn.clicked.connect(lambda: self.set_forecast_digit("小"))
        forecast_row3.addWidget(self.digit_small_btn)

        forecast_layout.addLayout(forecast_row3)

        forecast_scroll.setWidget(forecast_group)
        forecast_outer_layout.addWidget(forecast_scroll)
        predict_splitter.addWidget(forecast_widget)
        predict_splitter.setSizes([150, 200])  # 设置初始比例

        left_main_splitter.addWidget(predict_splitter)
        left_main_splitter.setSizes([100, 200, 350])  # 设置初始比例

        layout.addWidget(left_main_splitter, 1)

        return widget

    def create_right_panel(self) -> QWidget:
        """创建右侧面板"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # 标题
        title_label = QLabel("📜 历史记录")
        title_label.setStyleSheet(f"""
            QLabel {{
                font-size: 16px;
                font-weight: bold;
                color: {Config.COLORS['primary']};
            }}
        """)
        layout.addWidget(title_label)

        # 操作按钮
        btn_row = QHBoxLayout()

        add_btn = ToolbarButton("➕ 添加", Config.COLORS['success'])
        add_btn.clicked.connect(self.add_record)
        btn_row.addWidget(add_btn)

        edit_btn = ToolbarButton("✏️ 编辑", Config.COLORS['info'])
        edit_btn.clicked.connect(self.edit_record)
        btn_row.addWidget(edit_btn)

        delete_btn = ToolbarButton("🗑️ 删除", Config.COLORS['danger'])
        delete_btn.clicked.connect(self.delete_record)
        btn_row.addWidget(delete_btn)

        layout.addLayout(btn_row)

        btn_row2 = QHBoxLayout()

        batch_add_btn = ToolbarButton("📦 批量添加", Config.COLORS['primary'])
        batch_add_btn.clicked.connect(self.batch_add_records)
        btn_row2.addWidget(batch_add_btn)

        batch_delete_btn = ToolbarButton("🧹 批量删除", Config.COLORS['warning'])
        batch_delete_btn.clicked.connect(self.batch_delete_records)
        btn_row2.addWidget(batch_delete_btn)

        clear_all_btn = ToolbarButton("⚠️ 清空所有", Config.COLORS['danger'])
        clear_all_btn.setProperty("class", "danger")
        clear_all_btn.clicked.connect(self.clear_all_history)
        btn_row2.addWidget(clear_all_btn)

        layout.addLayout(btn_row2)

        # 历史记录与数据统计分隔条
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {Config.COLORS['border']};
                height: 6px;
                margin: 2px 0px;
                border-radius: 3px;
            }}
            QSplitter::handle:hover {{
                background-color: {Config.COLORS['primary']};
            }}
        """)

        # 历史记录表格 - 添加滚动条
        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        history_layout.setContentsMargins(0, 0, 0, 0)

        self.history_scroll = QScrollArea()
        self.history_scroll.setWidgetResizable(True)
        self.history_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {Config.COLORS['border']};
                border-radius: 6px;
                background-color: white;
            }}
            QScrollBar:vertical {{
                background-color: {Config.COLORS['white']};
                width: 12px;
                border-radius: 6px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {Config.COLORS['primary']};
                min-height: 30px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {Config.COLORS['primary_dark']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background-color: {Config.COLORS['white']};
            }}
        """)

        self.history_table = HistoryTable()
        self.history_table.itemSelectionChanged.connect(self.on_table_selection_changed)
        self.history_scroll.setWidget(self.history_table)
        history_layout.addWidget(self.history_scroll)

        right_splitter.addWidget(history_widget)

        # 统计信息 - 添加滚动条
        stats_widget = QWidget()
        stats_outer_layout = QVBoxLayout(stats_widget)
        stats_outer_layout.setContentsMargins(0, 0, 0, 0)

        self.stats_scroll = QScrollArea()
        self.stats_scroll.setWidgetResizable(True)
        self.stats_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {Config.COLORS['border']};
                border-radius: 6px;
                background-color: white;
            }}
            QScrollBar:vertical {{
                background-color: {Config.COLORS['white']};
                width: 12px;
                border-radius: 6px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {Config.COLORS['success']};
                min-height: 30px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {Config.COLORS['primary_dark']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background-color: {Config.COLORS['white']};
            }}
        """)

        stats_group = QGroupBox("📈 数据统计")
        stats_layout = QVBoxLayout(stats_group)

        self.stats_label = QLabel("暂无数据")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)

        self.stats_scroll.setWidget(stats_group)
        stats_outer_layout.addWidget(self.stats_scroll)

        right_splitter.addWidget(stats_widget)
        right_splitter.setSizes([300, 150])  # 设置初始比例

        layout.addWidget(right_splitter, 1)

        return widget

    def on_algorithm_changed(self, index: int):
        """算法改变时的处理"""
        self.main_window.data_manager.update_settings('last_algorithm', index)

    def on_number_clicked(self, number: int):
        """数字按钮点击"""
        btn = self.number_buttons[number]
        btn.set_selected(not btn.is_selected)

    def set_forecast_size(self, size: str):
        """设置大小预判"""
        self.size_forecast.setText(size)
        self.size_forecast.setStyleSheet(f"""
            QLabel {{
                color: {Config.COLORS['primary']};
                font-weight: bold;
            }}
        """)

    def set_forecast_odd(self, odd: str):
        """设置单双预判"""
        self.odd_forecast.setText(odd)
        self.odd_forecast.setStyleSheet(f"""
            QLabel {{
                color: {Config.COLORS['purple']};
                font-weight: bold;
            }}
        """)

    def set_forecast_digit(self, digit: str):
        """设置尾数预判"""
        self.digit_forecast.setText(digit)
        self.digit_forecast.setStyleSheet(f"""
            QLabel {{
                color: {Config.COLORS['warning']};
                font-weight: bold;
            }}
        """)

    def predict(self):
        """执行预测"""
        history = self.data_manager.get_history()
        algo_index = self.algorithm_combo.currentIndex()

        # 创建算法实例
        self.algorithm = PredictionAlgorithms(history)

        # 执行预测
        main_numbers, special = self.algorithm.predict(algo_index)

        # 保存预测结果
        self.current_prediction = {
            'numbers': main_numbers,
            'special': special,
            'algorithm': Config.ALGORITHMS[algo_index]
        }

        # 更新UI
        self.display_prediction(main_numbers, special)
        self.update_number_buttons(main_numbers, special)
        self.update_forecast(special)

    def extract(self):
        """随机抽取"""
        if not self.current_prediction:
            # 如果没有预测，先预测
            self.predict()
            return

        # 使用预测结果
        history = self.data_manager.get_history()
        if not history:
            # 无历史数据时纯随机
            main_numbers = random.sample(range(1, 50), 6)
            special = random.choice([n for n in range(1, 50) if n not in main_numbers])
        else:
            # 使用历史数据调整
            main_numbers = self.current_prediction['numbers'].copy()
            special = self.current_prediction['special']

            # 添加一些随机性
            if random.random() < 0.3:
                idx = random.randint(0, 5)
                available = [n for n in range(1, 50) if n not in main_numbers]
                if available:
                    main_numbers[idx] = random.choice(available)

            if random.random() < 0.2:
                available = [n for n in range(1, 50) if n not in main_numbers]
                if available:
                    special = random.choice(available)

        # 更新显示
        self.display_prediction(main_numbers, special, is_final=True)
        self.update_number_buttons(main_numbers, special)

    def display_prediction(self, numbers: List[int], special: int, is_final: bool = False):
        """显示预测结果"""
        numbers_str = ' '.join(
            f"<span style='color: {Config.COLORS['primary']}; font-weight: bold;'>{str(n).zfill(2)}</span>" for n in
            numbers)
        special_str = f"<span style='color: {Config.COLORS['danger']}; font-weight: bold;'>{str(special).zfill(2)}</span>"

        title = "🎯 最终抽取结果" if is_final else "🔮 预测结果"
        title_color = Config.COLORS['success'] if is_final else Config.COLORS['primary']

        result_html = f'''
        <div style="text-align: center;">
            <div style="font-size: 14px; color: {title_color}; font-weight: bold; margin-bottom: 10px;">
                {title}
            </div>
            <div style="font-size: 18px; margin-bottom: 10px;">
                主号码: {numbers_str}
            </div>
            <div style="font-size: 18px;">
                特别号: {special_str}
            </div>
        </div>
        '''
        self.result_label.setText(result_html)
        self.result_label.setTextFormat(Qt.TextFormat.RichText)

    def update_number_buttons(self, numbers: List[int], special: int):
        """更新数字按钮状态"""
        for num, btn in self.number_buttons.items():
            if num in numbers:
                btn.set_selected(True)
            elif num == special:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {Config.COLORS['danger']};
                        color: {Config.COLORS['white']};
                        border: 2px solid #D32F2F;
                        border-radius: 25px;
                        font-size: 16px;
                        font-weight: bold;
                    }}
                """)
            else:
                btn.set_selected(False)

    def update_forecast(self, special: int):
        """更新预判"""
        # 大小
        size = "大 (>24)" if special > 24 else "小 (≤24)"
        self.set_forecast_size(size)

        # 单双
        odd = "单" if special % 2 == 1 else "双"
        self.set_forecast_odd(odd)

        # 尾数
        digit = special % 10
        digit_size = "大 (5-9)" if digit >= 5 else "小 (0-4)"
        self.set_forecast_digit(digit_size)

    def load_history(self):
        """加载历史记录"""
        history = self.data_manager.get_history()
        self.update_table(history)
        self.update_statistics()

    def update_table(self, history: List[Dict[str, Any]]):
        """更新表格"""
        self.history_table.setRowCount(0)

        for i, record in enumerate(history):
            self.history_table.insertRow(i)

            # 期号
            period_item = QTableWidgetItem(f"第{record.get('period', '?')}期")
            self.history_table.setItem(i, 0, period_item)

            # 日期
            date_item = QTableWidgetItem(record.get('date', '?'))
            self.history_table.setItem(i, 1, date_item)

            # 主号码
            numbers = record.get('numbers', [])
            numbers_str = ' '.join(str(n).zfill(2) for n in numbers)
            numbers_item = QTableWidgetItem(numbers_str)
            numbers_item.setForeground(QColor(Config.COLORS['primary']))
            self.history_table.setItem(i, 2, numbers_item)

            # 特别号
            special = record.get('special', 0)
            special_item = QTableWidgetItem(str(special).zfill(2))
            special_item.setForeground(QColor(Config.COLORS['danger']))
            self.history_table.setItem(i, 3, special_item)

            # 操作按钮
            self.history_table.setCellWidget(i, 4, self.create_action_buttons(i))

    def create_action_buttons(self, row: int) -> QWidget:
        """创建操作按钮"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)

        edit_btn = QPushButton("编辑")
        edit_btn.setFixedSize(40, 25)
        edit_btn.clicked.connect(lambda: self.edit_row(row))
        layout.addWidget(edit_btn)

        del_btn = QPushButton("删除")
        del_btn.setFixedSize(40, 25)
        del_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Config.COLORS['danger']};
                color: white;
                border: none;
                border-radius: 3px;
            }}
        """)
        del_btn.clicked.connect(lambda: self.delete_row(row))
        layout.addWidget(del_btn)

        return widget

    def edit_row(self, row: int):
        """编辑行"""
        history = self.data_manager.get_history()
        if 0 <= row < len(history):
            record = history[row]
            self.show_edit_dialog(record, row)

    def delete_row(self, row: int):
        """删除行"""
        self.data_manager.delete_history([row])
        self.load_history()
        self.main_window.show_status_message("记录已删除")

    def on_table_selection_changed(self):
        """表格选择改变"""
        pass

    def add_record(self):
        """添加记录"""
        self.show_edit_dialog(None, -1)

    def edit_record(self):
        """编辑选中记录"""
        selected = self.history_table.selectedIndexes()
        if selected:
            row = selected[0].row()
            self.edit_row(row)
        else:
            self.main_window.show_status_message("请先选择要编辑的记录")

    def delete_record(self):
        """删除选中记录"""
        selected_rows = set(index.row() for index in self.history_table.selectedIndexes())
        if selected_rows:
            self.data_manager.delete_history(sorted(selected_rows, reverse=True))
            self.load_history()
            self.main_window.show_status_message("记录已删除")
        else:
            self.main_window.show_status_message("请先选择要删除的记录")

    def batch_add_records(self):
        """批量添加记录"""
        dialog = BatchAddDialog(self)
        if dialog.exec():
            records = dialog.get_records()
            for record in records:
                self.data_manager.add_history(record)
            self.load_history()
            self.main_window.show_status_message(f"已添加 {len(records)} 条记录")

    def batch_delete_records(self):
        """批量删除记录"""
        selected_rows = set(index.row() for index in self.history_table.selectedIndexes())
        if selected_rows:
            reply = QMessageBox.question(
                self,
                "确认批量删除",
                f"确定要删除选中的 {len(selected_rows)} 条记录吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.data_manager.delete_history(sorted(selected_rows, reverse=True))
                self.load_history()
                self.main_window.show_status_message("记录已批量删除")
        else:
            self.main_window.show_status_message("请先选择要删除的记录")

    def clear_all_history(self):
        """清空所有历史"""
        reply = QMessageBox.warning(
            self,
            "⚠️ 确认清空",
            "确定要清空所有历史记录吗？此操作不可恢复！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.data_manager.clear_history()
            self.load_history()
            self.main_window.show_status_message("所有历史记录已清空")

    def show_edit_dialog(self, record: Optional[Dict], index: int):
        """显示编辑对话框"""
        dialog = EditRecordDialog(record, self)
        if dialog.exec():
            new_record = dialog.get_record()
            if index >= 0:
                self.data_manager.update_history(index, new_record)
            else:
                self.data_manager.add_history(new_record)
            self.load_history()
            status = "更新" if index >= 0 else "添加"
            self.main_window.show_status_message(f"记录{status}成功")

    def update_statistics(self):
        """更新统计信息"""
        history = self.data_manager.get_history()

        if not history:
            self.stats_label.setText("暂无数据")
            return

        # 统计各数字出现频率
        all_numbers = []
        for record in history:
            all_numbers.extend(record.get('numbers', []))
            if 'special' in record:
                all_numbers.append(record['special'])

        counter = Counter(all_numbers)
        total = len(all_numbers)

        # 找出最热和最冷的数字
        if counter:
            hot = counter.most_common(5)
            cold = counter.most_common(49)[-5:][::-1]

            hot_str = ', '.join([f"{n}({c})" for n, c in hot])
            cold_str = ', '.join([f"{n}({c})" for n, c in cold])

            stats_html = f'''
            <div>
                <b>总记录数:</b> {len(history)} 条<br/>
                <b>总开奖数:</b> {total} 个<br/>
                <b>最热号码:</b> <span style="color: {Config.COLORS['danger']};">{hot_str}</span><br/>
                <b>最冷号码:</b> <span style="color: {Config.COLORS['info']};">{cold_str}</span>
            </div>
            '''
            self.stats_label.setText(stats_html)
            self.stats_label.setTextFormat(Qt.TextFormat.RichText)


# ============================================================================
# 编辑记录对话框
# ============================================================================

class EditRecordDialog(QDialog):
    """编辑记录对话框"""

    def __init__(self, record: Optional[Dict], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.record = record
        self.init_ui()

        if record:
            self.load_record()

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("编辑记录" if self.record else "添加记录")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # 期号
        period_layout = QHBoxLayout()
        period_layout.addWidget(QLabel("期号:"))
        self.period_edit = QLineEdit()
        self.period_edit.setPlaceholderText("例如: 116")
        period_layout.addWidget(self.period_edit)
        layout.addLayout(period_layout)

        # 日期
        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("日期:"))
        self.date_edit = QLineEdit()
        self.date_edit.setPlaceholderText("例如: 2026-04-26")
        date_layout.addWidget(self.date_edit)
        layout.addLayout(date_layout)

        # 主号码
        numbers_layout = QHBoxLayout()
        numbers_layout.addWidget(QLabel("主号码:"))
        self.numbers_edit = QLineEdit()
        self.numbers_edit.setPlaceholderText("例如: 15 46 16 10 48 33")
        numbers_layout.addWidget(self.numbers_edit)
        layout.addLayout(numbers_layout)

        # 特别号
        special_layout = QHBoxLayout()
        special_layout.addWidget(QLabel("特别号:"))
        self.special_edit = QLineEdit()
        self.special_edit.setPlaceholderText("例如: 22")
        special_layout.addWidget(self.special_edit)
        layout.addLayout(special_layout)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

    def load_record(self):
        """加载记录"""
        if not self.record:
            return

        self.period_edit.setText(str(self.record.get('period', '')))
        self.date_edit.setText(str(self.record.get('date', '')))
        numbers = self.record.get('numbers', [])
        self.numbers_edit.setText(' '.join(str(n) for n in numbers))
        self.special_edit.setText(str(self.record.get('special', '')))

    def get_record(self) -> Dict[str, Any]:
        """获取记录"""
        numbers = [int(n) for n in self.numbers_edit.text().split() if n.isdigit()]
        return {
            'period': self.period_edit.text() or '未知',
            'date': self.date_edit.text() or datetime.now().strftime('%Y-%m-%d'),
            'numbers': numbers[:6],
            'special': int(self.special_edit.text()) if self.special_edit.text().isdigit() else 0
        }


# ============================================================================
# 批量添加对话框
# ============================================================================

class BatchAddDialog(QDialog):
    """批量添加对话框"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("批量添加记录")
        self.setMinimumSize(500, 400)

        layout = QVBoxLayout(self)

        # 说明
        help_label = QLabel(
            "每行一条记录，格式：\n"
            "期号 日期 主号码(空格分隔) 特别号\n"
            "例如: 116 2026-04-26 15 46 16 10 48 33 22"
        )
        layout.addWidget(help_label)

        # 文本框
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText(
            "每行一条记录，例如：\n"
            "116 2026-04-26 15 46 16 10 48 33 22\n"
            "115 2026-04-25 08 23 35 41 12 44 19\n"
            "..."
        )
        layout.addWidget(self.text_edit, 1)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

    def get_records(self) -> List[Dict[str, Any]]:
        """获取记录列表"""
        records = []
        lines = self.text_edit.toPlainText().strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 9:
                continue

            try:
                period = parts[0]
                date = parts[1]
                numbers = [int(parts[i]) for i in range(2, 8) if parts[i].isdigit()]
                special = int(parts[8]) if parts[8].isdigit() else 0

                if len(numbers) == 6 and 1 <= special <= 49:
                    records.append({
                        'period': period,
                        'date': date,
                        'numbers': numbers,
                        'special': special
                    })
            except (ValueError, IndexError):
                continue

        return records


# ============================================================================
# 字体设置对话框
# ============================================================================

class FontSettingsDialog(QDialog):
    """字体设置对话框"""

    def __init__(self, current_size: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.current_size = current_size
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("字体设置")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # 字体大小
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("字体大小:"))

        self.size_combo = QComboBox()
        font_names = list(Config.FONT_SIZES.keys())
        self.size_combo.addItems(font_names)

        # 设置当前值
        current_name = None
        for name, size in Config.FONT_SIZES.items():
            if size == self.current_size:
                current_name = name
                break

        if current_name:
            self.size_combo.setCurrentText(current_name)

        size_layout.addWidget(self.size_combo)
        layout.addLayout(size_layout)

        # 预览
        preview_label = QLabel("预览文本 - AaBbCcDd 1234567890")
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_label.setStyleSheet(f"""
            QLabel {{
                background-color: {Config.COLORS['primary_light']};
                border: 1px solid {Config.COLORS['primary']};
                border-radius: 4px;
                padding: 20px;
                margin: 10px;
            }}
        """)
        layout.addWidget(preview_label)

        # 更新预览
        def update_preview():
            size_name = self.size_combo.currentText()
            size = Config.FONT_SIZES.get(size_name, 12)
            preview_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {Config.COLORS['primary_light']};
                    border: 1px solid {Config.COLORS['primary']};
                    border-radius: 4px;
                    padding: 20px;
                    margin: 10px;
                    font-size: {size}px;
                }}
            """)

        self.size_combo.currentTextChanged.connect(update_preview)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

    def get_font_size(self) -> int:
        """获取字体大小"""
        size_name = self.size_combo.currentText()
        return Config.FONT_SIZES.get(size_name, 12)


# ============================================================================
# 主窗口类
# ============================================================================

class MainWindow(QMainWindow):
    """
    主窗口类
    包含所有核心功能
    """

    def __init__(self):
        super().__init__()

        # 数据管理器
        self.data_manager = DataManager()

        # 初始化UI
        self.init_ui()

        # 加载设置
        self.load_settings()

        # 信号连接
        self.connect_signals()

    def init_ui(self):
        """初始化UI"""
        # 窗口基本设置
        self.setWindowTitle(Config.WINDOW_TITLE)
        self.setMinimumSize(Config.WINDOW_MIN_WIDTH, Config.WINDOW_MIN_HEIGHT)
        self.setStyleSheet(StyleSheet.get_stylesheet())

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 顶部工具栏
        self.create_toolbar()

        # 标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(DataConversionPanel(self), "📋 数据转换")
        self.tab_widget.addTab(AnalysisPanel(self), "🎲 分析预测")

        main_layout.addWidget(self.tab_widget, 1)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.show_status_message("就绪")

    def create_toolbar(self):
        """创建工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # 导入按钮
        import_btn = ToolbarButton("📂 导入", Config.COLORS['primary'])
        import_btn.clicked.connect(self.import_data)
        toolbar.addWidget(import_btn)

        # 导出按钮
        export_btn = ToolbarButton("📤 导出", Config.COLORS['success'])
        export_btn.clicked.connect(self.export_data)
        toolbar.addWidget(export_btn)

        toolbar.addSeparator()

        # 保存按钮
        save_btn = ToolbarButton("💾 保存", Config.COLORS['info'])
        save_btn.clicked.connect(self.save_current_data)
        toolbar.addWidget(save_btn)

        toolbar.addSeparator()

        # 字体设置
        font_label = QLabel("字体:")
        toolbar.addWidget(font_label)

        self.font_combo = QComboBox()
        self.font_combo.addItems(list(Config.FONT_SIZES.keys()))
        self.font_combo.currentIndexChanged.connect(self.on_font_changed)
        toolbar.addWidget(self.font_combo)

        toolbar.addSeparator()

        # 关于按钮
        about_btn = ToolbarButton("ℹ️ 关于", Config.COLORS['purple'])
        about_btn.clicked.connect(self.show_about)
        toolbar.addWidget(about_btn)

    def connect_signals(self):
        """连接信号"""
        pass

    def load_settings(self):
        """加载设置"""
        settings = self.data_manager.get_settings()

        # 字体大小
        font_index = settings.get('font_size', Config.DEFAULT_FONT_INDEX)
        font_names = list(Config.FONT_SIZES.keys())
        if 0 <= font_index < len(font_names):
            self.font_combo.setCurrentIndex(font_index)

    def on_font_changed(self, index: int):
        """字体改变"""
        font_size = Config.FONT_SIZES.get(list(Config.FONT_SIZES.keys())[index], 12)
        self.data_manager.update_settings('font_size', index)

        # 更新样式表
        self.setStyleSheet(StyleSheet.get_stylesheet(font_size=font_size))
        self.show_status_message(f"字体已更改为 {list(Config.FONT_SIZES.keys())[index]}")

    def import_data(self):
        """导入数据"""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "导入数据",
            "",
            "JSON文件 (*.json);;所有文件 (*)"
        )

        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    imported_data = json.load(f)

                if 'history' in imported_data:
                    # 合并数据
                    current_history = self.data_manager.get_history()
                    new_history = imported_data['history']

                    # 去重
                    existing_keys = set()
                    for record in current_history:
                        key = f"{record.get('period')}_{record.get('date')}"
                        existing_keys.add(key)

                    for record in new_history:
                        key = f"{record.get('period')}_{record.get('date')}"
                        if key not in existing_keys:
                            self.data_manager.add_history(record)

                    self.show_status_message(f"成功导入 {len(new_history)} 条记录")
                    QMessageBox.information(self, "导入成功", f"成功导入 {len(new_history)} 条记录")
                else:
                    QMessageBox.warning(self, "导入失败", "无效的数据格式")

            except Exception as e:
                QMessageBox.critical(self, "导入失败", f"导入失败: {str(e)}")

    def export_data(self):
        """导出数据"""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "导出数据",
            f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON文件 (*.json);;所有文件 (*)"
        )

        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.data_manager.data, f, ensure_ascii=False, indent=2)

                self.show_status_message("数据导出成功")
                QMessageBox.information(self, "导出成功", "数据导出成功！")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"导出失败: {str(e)}")

    def save_current_data(self):
        """保存当前数据"""
        if self.data_manager.save_data():
            self.show_status_message("数据保存成功")
            QMessageBox.information(self, "保存成功", "数据保存成功！")
        else:
            self.show_status_message("数据保存失败")
            QMessageBox.warning(self, "保存失败", "数据保存失败！")

    def add_history_record(self, record: Dict[str, Any]):
        """添加历史记录"""
        self.data_manager.add_history(record)

        # 如果在分析面板，刷新显示
        if self.tab_widget.currentIndex() == 1:
            analysis_panel = self.tab_widget.widget(1)
            if hasattr(analysis_panel, 'load_history'):
                analysis_panel.load_history()

    def show_status_message(self, message: str):
        """显示状态栏消息"""
        self.status_bar.showMessage(message, 3000)

    def show_about(self):
        """显示关于对话框"""
        about_text = f"""
        <h2>{Config.WINDOW_TITLE}</h2>
        <p>一个基于PyQt6开发的完整数据分析与预测系统</p>
        <p><b>版本:</b> 1.0.0</p>
        <p><b>技术栈:</b></p>
        <ul>
            <li>PyQt6 - GUI框架</li>
            <li>NumPy - 数值计算{f'<span style="color: green;"> ✓</span>' if HAS_NUMPY else '<span style="color: red;"> ✗</span>'}</li>
            <li>Pandas - 数据处理{f'<span style="color: green;"> ✓</span>' if HAS_PANDAS else '<span style="color: red;"> ✗</span>'}</li>
            <li>Matplotlib - 绑图{f'<span style="color: green;"> ✓</span>' if HAS_MATPLOTLIB else '<span style="color: red;"> ✗</span>'}</li>
            <li>SciPy - 科学计算{f'<span style="color: green;"> ✓</span>' if HAS_SCIPY else '<span style="color: red;"> ✗</span>'}</li>
            <li>Statsmodels - 统计模型{f'<span style="color: green;"> ✓</span>' if HAS_STATSMODELS else '<span style="color: red;"> ✗</span>'}</li>
            <li>Scikit-learn - 机器学习{f'<span style="color: green;"> ✓</span>' if HAS_SKLEARN else '<span style="color: red;"> ✗</span>'}</li>
            <li>Optuna - 超参数优化{f'<span style="color: green;"> ✓</span>' if HAS_OPTUNA else '<span style="color: red;"> ✗</span>'}</li>
            <li>PyTorch - 深度学习{f'<span style="color: green;"> ✓</span>' if HAS_TORCH else '<span style="color: red;"> ✗</span>'}</li>
        </ul>
        <p><b>功能:</b></p>
        <ul>
            <li>数据格式转换</li>
            <li>49个数字随机抽取</li>
            <li>12种预测算法</li>
            <li>历史记录管理</li>
        </ul>
        """
        QMessageBox.about(self, "关于", about_text)


# ============================================================================
# 程序入口
# ============================================================================

def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用信息
    app.setApplicationName(Config.WINDOW_TITLE)
    app.setOrganizationName("DataAnalysis")

    # 创建并显示主窗口
    window = MainWindow()
    window.show()

    # 运行应用
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

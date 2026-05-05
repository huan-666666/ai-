"""
PyQt6 综合工具 - 数字排序 + 数字选尾
功能：全局字体调节、页边距间距调节、自适应布局、拖拽分割
界面：纯白底色 + 自定义纯色UI、无暗色系、无默认灰
"""

# ===================== 所用库清单 =====================
# PyQt6.QtWidgets ：所有界面控件（窗口、标签、按钮、输入框、标签页等）
# PyQt6.QtGui    ：字体、颜色配置
# PyQt6.QtCore   ：布局、尺寸策略、全局配置
# ======================================================

from PyQt6.QtWidgets import *
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtCore import Qt

# ===================== 全局配置 =====================
# 界面配色（纯白底色 + 纯色高亮，禁用所有灰色/暗色）
COLOR_BG = "#FFFFFF"                 # 全局纯白背景
COLOR_PRIMARY = "#BA94E8"            # 主色（紫色）
COLOR_SUCCESS = "#2ECC71"            # 成功绿
COLOR_DANGER = "#E74C3C"             # 失败红
COLOR_CARD = "#F8F9FA"               # 区块浅白底
COLOR_BORDER = "#FFD1DC"             # 边框色

# 字体大小对应：初号 ~ 小四号
FONT_SIZES = {
    "初号": 42,
    "小初": 36,
    "一号": 26,
    "二号": 22,
    "三号": 18,
    "四号": 14,
    "小四号": 12
}

# ===================== 主窗口 =====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("数字综合工具")
        self.setGeometry(100, 100, 1100, 700)
        self.setStyleSheet(f"background-color: {COLOR_BG};")

        # 全局变量
        self.global_font_size = FONT_SIZES["三号"]  # 默认字体
        self.global_margin = 16                     # 页边距
        self.global_spacing = 12                    # 控件间距

        # 主容器 + 布局
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)
        self.main_layout.setContentsMargins(self.global_margin, self.global_margin, self.global_margin, self.global_margin)
        self.main_layout.setSpacing(self.global_spacing)

        # 顶部控制栏 + 标签页 + 拖拽布局
        self.create_top_bar()
        self.create_tab_and_splitter()

        # 初始刷新界面
        self.update_all_styles()

    # ===================== 顶部全局控制栏 =====================
    def create_top_bar(self):
        bar = QHBoxLayout()
        bar.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # 字体调节
        self.font_combo = QComboBox()
        self.font_combo.addItems(FONT_SIZES.keys())
        self.font_combo.setCurrentText("三号")
        self.font_combo.currentTextChanged.connect(self.change_global_font)

        # 页边距调节
        self.margin_slider = QSlider(Qt.Orientation.Horizontal)
        self.margin_slider.setRange(8, 40)
        self.margin_slider.setValue(16)
        self.margin_slider.valueChanged.connect(self.change_margin)

        # 间距调节
        self.spacing_slider = QSlider(Qt.Orientation.Horizontal)
        self.spacing_slider.setRange(4, 24)
        self.spacing_slider.setValue(12)
        self.spacing_slider.valueChanged.connect(self.change_spacing)

        # 加入标题
        bar.addWidget(QLabel("全局字体："))
        bar.addWidget(self.font_combo)
        bar.addWidget(QLabel(" 页边距："))
        bar.addWidget(self.margin_slider)
        bar.addWidget(QLabel(" 区块间距："))
        bar.addWidget(self.spacing_slider)

        self.main_layout.addLayout(bar)

    # ===================== 标签页 + 拖拽布局 =====================
    def create_tab_and_splitter(self):
        self.tab = QTabWidget()
        self.tab.setStyleSheet(f"""
            QTabWidget::pane {{border:1px solid {COLOR_BORDER}; background:{COLOR_CARD};}}
            QTabBar::tab {{padding:8px 16px; background:{COLOR_CARD};}}
            QTabBar::tab:selected {{background:{COLOR_PRIMARY}; color:white;}}
        """)

        # 两个功能页面
        self.tab.addTab(NumberSortTool(), "数字排序工具")
        self.tab.addTab(NumberTailTool(), "数字选尾工具")

        # 拖拽分割器（VS Code 逻辑）
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.tab)
        splitter.setStyleSheet("QSplitter::handle{background-color: #FFD1DC; width:6px;}")
        self.main_layout.addWidget(splitter)

    # ===================== 全局样式刷新 =====================
    def update_all_styles(self):
        self.setStyleSheet(f"background-color: {COLOR_BG};")
        self.main_layout.setContentsMargins(self.global_margin, self.global_margin, self.global_margin, self.global_margin)
        self.main_layout.setSpacing(self.global_spacing)

        font = QFont()
        font.setPointSize(self.global_font_size)
        self.setFont(font)

        for widget in self.findChildren(QWidget):
            widget.setFont(font)

    # ===================== 全局字体切换 =====================
    def change_global_font(self, name):
        self.global_font_size = FONT_SIZES[name]
        self.update_all_styles()

    # ===================== 页边距调节 =====================
    def change_margin(self, value):
        self.global_margin = value
        self.update_all_styles()

    # ===================== 间距调节 =====================
    def change_spacing(self, value):
        self.global_spacing = value
        self.update_all_styles()

# ===================== 功能1：数字排序工具 =====================
class NumberSortTool(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 输入框
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("请输入数字，支持逗号/句号/空格分隔，例如：1, 5。8 12 33")
        layout.addWidget(QLabel("输入数字："))
        layout.addWidget(self.input_edit)

        # 执行按钮
        btn = QPushButton("开始排序分析")
        btn.setStyleSheet(f"background:{COLOR_SUCCESS}; color:white; padding:8px;")
        layout.addWidget(btn)

        # 结果展示
        self.result = QTextEdit()
        self.result.setReadOnly(True)
        layout.addWidget(QLabel("分析结果："))
        layout.addWidget(self.result)

        btn.clicked.connect(self.analyze)

    def analyze(self):
        text = self.input_edit.toPlainText()
        # 替换所有分隔符
        for c in "，。、；：()[]":
            text = text.replace(c, ",")
        items = text.replace(" ", ",").split(",")
        nums = []

        # 提取有效数字
        for item in items:
            item = item.strip()
            if item.isdigit():
                num = int(item)
                if 1 <= num <= 49:
                    nums.append(num)

        # 去重 + 排序
        unique = sorted(list(set(nums)))
        # 统计重复
        count = {n: nums.count(n) for n in nums}
        repeat = {k:v for k,v in count.items() if v>1}
        # 查缺失
        missing = [n for n in range(1,50) if n not in nums]

        # 显示结果
        res = f"【去重排序】{unique}\n\n"
        res += f"【重复统计】{repeat if repeat else '无重复'}\n\n"
        res += f"【缺失数字】{missing}"
        self.result.setText(res)

# ===================== 功能2：数字选尾工具 =====================
class NumberTailTool(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 数字输入
        self.num_edit = QTextEdit()
        self.num_edit.setPlaceholderText("格式：01,02,03,15,28,49")
        layout.addWidget(QLabel("输入全部数字："))
        layout.addWidget(self.num_edit)

        # 尾数输入
        self.tail_edit = QLineEdit()
        self.tail_edit.setPlaceholderText("输入尾数，例如：1,3,5,7,9")
        layout.addWidget(QLabel("输入要筛选的尾数："))
        layout.addWidget(self.tail_edit)

        # 执行按钮
        btn = QPushButton("筛选符合尾数的数字")
        btn.setStyleSheet(f"background:{COLOR_PRIMARY}; color:white; padding:8px;")
        layout.addWidget(btn)

        # 结果
        self.result = QTextEdit()
        self.result.setReadOnly(True)
        layout.addWidget(QLabel("筛选结果："))
        layout.addWidget(self.result)

        btn.clicked.connect(self.filter_tail)

    def filter_tail(self):
        nums_text = self.num_edit.toPlainText()
        tail_text = self.tail_edit.text()

        # 提取数字
        nums = [int(n.strip()) for n in nums_text.replace(" ","").split(",") if n.strip().isdigit()]
        tails = [t.strip() for t in tail_text.replace(" ","").split(",") if t.strip()]

        # 筛选尾数
        match = []
        for num in nums:
            if 1<=num<=49 and str(num)[-1] in tails:
                match.append(num)
        match = sorted(match)
        self.result.setText(f"【筛选结果】{match}")

# ===================== 启动程序 =====================
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
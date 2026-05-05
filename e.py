import customtkinter as ctk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("六合彩分析工具")
app.geometry("700x500")

# 标题
title = ctk.CTkLabel(app, text="六合彩数据分析", font=("微软雅黑", 24))
title.pack(pady=20)

# 输入期号
entry = ctk.CTkEntry(app, placeholder_text="输入开奖期号", width=366)
entry.pack(pady=10)

# 查询按钮
def search():
    num = entry.get()
    title.configure(text=f"正在查询：第{num}期")

search_btn = ctk.CTkButton(app, text="开始查询", command=search)
search_btn.pack(pady=10)

# 结果文本框
result_box = ctk.CTkTextbox(app, width=600, height=250)
result_box.pack(pady=20)

app.mainloop()
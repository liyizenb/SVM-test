import tkinter as tk
from tkinter import messagebox
import joblib
import jieba

# 加载已保存的模型和向量器
clf = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# 情感分析函数
def analyze_sentiment(text):
    cut_text = " ".join(jieba.cut(text))
    tfidf = vectorizer.transform([cut_text])
    prediction = clf.predict(tfidf)[0]
    proba = None
    if hasattr(clf, "decision_function"):
        score = clf.decision_function(tfidf)[0]
        proba = 1 / (1 + pow(2.718, -score))  # Sigmoid approximation
    sentiment = "正向 😊" if prediction == 1 else "负向 😠"
    message = f"预测结果：{sentiment}"
    if proba is not None:
        message += f"\n置信度约为：{proba:.2%}"
    messagebox.showinfo("情感分析结果", message)

# 构建界面
def launch_gui():
    win = tk.Tk()
    win.title("网络评论情感分析器 - SVM模型")
    win.geometry("500x300")

    tk.Label(win, text="请输入一段评论文本：", font=("微软雅黑", 12)).pack(pady=10)
    text_entry = tk.Text(win, height=8, font=("微软雅黑", 11))
    text_entry.pack(padx=10)

    def on_submit():
        input_text = text_entry.get("1.0", tk.END).strip()
        if input_text:
            analyze_sentiment(input_text)
        else:
            messagebox.showwarning("输入为空", "请输入文本内容再分析")

    tk.Button(win, text="分析情感倾向", font=("微软雅黑", 12), command=on_submit).pack(pady=15)
    win.mainloop()

launch_gui()

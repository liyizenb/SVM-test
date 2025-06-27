import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import time
import joblib
from tqdm import tqdm

# 读取数据集
data = pd.read_csv("online_shoping.csv", encoding="utf-8-sig")

# 分词处理
def tokenize(text):
    return " ".join(jieba.cut(str(text)))

print("正在分词...")
data['cut_review'] = list(tqdm(map(tokenize, data['review']), total=len(data)))

# TF-IDF 特征提取
print("提取TF-IDF特征...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cut_review'])
y = data['label']

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型训练 + 时间统计
print("开始训练SVM模型...")
start_time = time.time()

clf = svm.SVC(kernel='linear', verbose=True)
clf.fit(X_train, y_train)

end_time = time.time()
print(f"\n训练完成，用时：{end_time - start_time:.2f} 秒")

# 保存模型和TF-IDF向量器
joblib.dump(clf, "svm_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("模型和向量器已保存为 'svm_model.pkl' 和 'tfidf_vectorizer.pkl'")

# 预测与评估
y_pred = clf.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, digits=4))

# 特征词可视化（如果是线性核）
if hasattr(clf, "coef_"):
    coefs = clf.coef_.toarray().flatten()
    top_n = 20
    top_indices = coefs.argsort()[-top_n:]
    words = [vectorizer.get_feature_names_out()[i] for i in top_indices]
    weights = coefs[top_indices]

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 6))
    plt.barh(words, weights, color="steelblue")
    plt.xlabel("SVM 权重")
    plt.title("影响情感分类的关键词汇（线性SVM）")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig("important_words.png")
    plt.show()
    print("重要特征词图像已保存为 'important_words.png'")

import pandas as pd
import jieba
import re
import time
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import joblib

# ------------------------------
# 1. 停用词 & 情感词典加载
# ------------------------------
with open("stopwords.txt", encoding="utf-8") as f:
    stopwords = set([line.strip() for line in f])

positive_words = {'喜欢', '满意', '棒', '好', '优秀', '值得', '快乐', '真理', '进步'}
negative_words = {'差', '糟糕', '坏', '失望', '低级', '失败', '一败涂地'}

# ------------------------------
# 2. 数据加载与处理
# ------------------------------
df = pd.read_csv("online_shoping.csv", encoding="utf-8-sig")
df['review'] = df['review'].fillna('').astype(str)

def clean_text(text):
    text = re.sub(r"[^\u4e00-\u9fa5]", " ", text)  # 仅保留中文
    words = jieba.cut(text)
    return " ".join(w for w in words if w not in stopwords and w.strip())

def emotion_features(text):
    tokens = text.split()
    pos_count = sum(1 for t in tokens if t in positive_words)
    neg_count = sum(1 for t in tokens if t in negative_words)
    return pd.Series([pos_count, neg_count, pos_count - neg_count])

print("正在清洗文本并提取特征...")
tqdm.pandas()
df["cut_review"] = df["review"].progress_apply(clean_text)
df[["pos_count", "neg_count", "emotion_score"]] = df["cut_review"].apply(emotion_features)

# ------------------------------
# 3. TF-IDF + 情感特征合并
# ------------------------------
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2, max_df=0.9)
X_tfidf = vectorizer.fit_transform(df["cut_review"])
X_dict = df[["pos_count", "neg_count", "emotion_score"]]
X = hstack([X_tfidf, X_dict.values])
y = df["label"]

# ------------------------------
# 4. 划分数据集
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------
# 5. 使用默认线性核 SVM 训练
# ------------------------------
print("开始训练 SVM...")
start_time = time.time()

clf = SVC(kernel="linear", C=1, gamma='scale')
clf.fit(X_train, y_train)

end_time = time.time()
print(f"训练完成，用时 {end_time - start_time:.2f} 秒")

# ------------------------------
# 6. 模型评估
# ------------------------------
y_pred = clf.predict(X_test)
print(f"\n准确率: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("混淆矩阵")
plt.show()

# ------------------------------
# 7. 模型保存
# ------------------------------
joblib.dump(clf, "svm_model_default.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer_default.pkl")
print("模型和向量器已保存为 'svm_model_default.pkl' 和 'tfidf_vectorizer_default.pkl'")

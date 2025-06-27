import pandas as pd
import jieba
import re
import time
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
import joblib

# ------------------------------ 停用词 & 情感词典加载 ------------------------------
with open("stopwords.txt", encoding="utf-8") as f:
    stopwords = set([line.strip() for line in f])

positive_words = {'喜欢', '满意', '棒', '好', '优秀', '值得', '快乐', '真理', '进步'}
negative_words = {'差', '糟糕', '坏', '失望', '低级', '失败', '一败涂地'}

# ------------------------------ 清洗函数 ------------------------------
def clean_text(text):
    text = re.sub(r"[^\u4e00-\u9fa5]", " ", text)
    words = jieba.cut(text)
    return " ".join(w for w in words if w not in stopwords and w.strip())

def emotion_features(text):
    tokens = text.split()
    pos_count = sum(1 for t in tokens if t in positive_words)
    neg_count = sum(1 for t in tokens if t in negative_words)
    return pd.Series([pos_count, neg_count, pos_count - neg_count])

# ------------------------------ 主流程 ------------------------------
df = pd.read_csv("online_shoping.csv", encoding="utf-8-sig")
df['review'] = df['review'].fillna('').astype(str)

tqdm.pandas()
df['cut_review'] = df['review'].progress_apply(clean_text)
df[['pos_count', 'neg_count', 'emotion_score']] = df['cut_review'].apply(emotion_features)

results = []

for cat in sorted(df['cat'].unique()):
    print(f"\n====== 正在处理类别：{cat} ======")
    sub_df = df[df['cat'] == cat]
    if sub_df['label'].nunique() < 2:
        print(f"类别“{cat}”不满足二分类要求，跳过")
        continue

    # 特征提取
    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2, max_df=0.9)
    X_tfidf = vectorizer.fit_transform(sub_df['cut_review'])
    X_dict = sub_df[['pos_count', 'neg_count', 'emotion_score']]
    X = hstack([X_tfidf, X_dict.values])
    y = sub_df['label']

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 训练模型
    start_time = time.time()
    clf = SVC(kernel="linear", C=1, gamma='scale')
    clf.fit(X_train, y_train)
    end_time = time.time()

    # 评估
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({'类别': cat, '样本数': len(sub_df), '准确率': f"{acc:.4f}", '训练耗时(s)': f"{end_time - start_time:.2f}"})

    # 保存模型（可选）
    joblib.dump(clf, f"svm_model_{cat}.pkl")
    joblib.dump(vectorizer, f"vectorizer_{cat}.pkl")

# ------------------------------ 输出准确率表格 ------------------------------
result_df = pd.DataFrame(results)
print("\n====== 各类别准确率汇总 ======")
print(result_df.to_markdown(index=False))

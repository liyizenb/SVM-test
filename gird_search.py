import pandas as pd
import jieba
import re
import time
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import joblib
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------
# 0. 随机抽样函数
# ------------------------------
def sample_csv(input_file, output_file, n, random_state=42):
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    sampled_df = df.sample(n=n, random_state=random_state)
    sampled_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 已从 {input_file} 中随机抽取 {n} 行，保存为 {output_file}")
    return sampled_df

# ------------------------------
# 1. 停用词 & 情感词典加载
# ------------------------------
with open("stopwords.txt", encoding="utf-8") as f:
    stopwords = set([line.strip() for line in f])

positive_words = {'喜欢', '满意', '棒', '好', '优秀', '值得', '快乐', '真理', '进步'}
negative_words = {'差', '糟糕', '坏', '失望', '低级', '失败', '一败涂地'}

# ------------------------------
# 2. 文本清洗与情感特征
# ------------------------------
def clean_text(text):
    text = re.sub(r"[^\u4e00-\u9fa5]", " ", text)
    words = jieba.cut(text)
    words = [w for w in words if w.strip() and w not in stopwords]
    return " ".join(words)

def emotion_features(text):
    tokens = text.split()
    pos_count = sum(1 for t in tokens if t in positive_words)
    neg_count = sum(1 for t in tokens if t in negative_words)
    return pd.Series([pos_count, neg_count, pos_count - neg_count])

# ------------------------------
# 主程序
# ------------------------------
def main():
    input_file = "online_shoping.csv"
    sampled_file = "sampled_data.csv"
    sample_size = 5000

    # Step 0: 抽样数据
    if not os.path.exists(sampled_file):
        df = sample_csv(input_file, sampled_file, n=sample_size)
    else:
        df = pd.read_csv(sampled_file, encoding='utf-8-sig')
        print(f"📄 已读取现有文件：{sampled_file}")

    df['review'] = df['review'].fillna('').astype(str)

    print("🚿 正在清洗文本...")
    tqdm.pandas()
    df["cut_review"] = df["review"].progress_apply(clean_text)

    print("🔍 提取情感词典特征...")
    df[["pos_count", "neg_count", "emotion_score"]] = df["cut_review"].apply(emotion_features)

    # Step 3: 特征提取
    print("📐 提取 TF-IDF 特征...")
    tfidf = TfidfVectorizer(
        max_features=8000,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2)
    )
    X_tfidf = tfidf.fit_transform(df['cut_review'])
    X_dict = df[['pos_count', 'neg_count', 'emotion_score']]
    X = hstack([X_tfidf, X_dict.values])
    y = df['label']

    # Step 4: 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Step 5: 模型训练与网格搜索
    print("🧠 训练模型中...")
    start_time = time.time()
    param_grid = {
        'C': [0.5, 1, 5],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    grid = GridSearchCV(SVC(), param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    end_time = time.time()

    clf = grid.best_estimator_
    print(f"✅ 训练完成！耗时 {end_time - start_time:.2f} 秒")
    print("📌 最优参数：", grid.best_params_)

    # Step 6: 模型评估
    y_pred = clf.predict(X_test)
    print(f"\n🎯 准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("混淆矩阵")
    plt.show()

    # Step 7: 模型保存
    joblib.dump(clf, "svm_model_with_lexicon.pkl")
    joblib.dump(tfidf, "tfidf_vectorizer_with_lexicon.pkl")
    print("💾 模型和向量器已保存为 'svm_model_with_lexicon.pkl' 和 'tfidf_vectorizer_with_lexicon.pkl'")

if __name__ == "__main__":
    main()

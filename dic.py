import pandas as pd
import jieba
from sklearn.metrics import accuracy_score

# 1. 加载数据
df = pd.read_csv("online_shoping.csv")
df['review'] = df['review'].fillna('').astype(str)

# 2. 构建词典（示例，建议用更全词典替换）
positive_words = {'喜欢', '满意', '棒', '好', '优秀', '值得', '美丽', '快乐', '自由', '真理', '进步'}
negative_words = {'差', '糟糕', '坏', '失望', '低级', '过于', '一败涂地'}
negation_words = {'不', '没', '无', '未', '否', '别'}
degree_words = {
    '非常': 2.0, '特别': 2.0, '十分': 2.0,
    '很': 1.5, '较': 1.2, '有点': 0.8, '稍微': 0.5
}

# 3. 情感分析函数
def analyze_sentiment(text):
    
    if pd.isnull(text):  # 如果为空
        return 2  # 视为中性
    text = str(text)  # 确保是字符串

    words = list(jieba.cut(text))
    score = 0
    i = 0
    while i < len(words):
        word = words[i]
        base = 0
        if word in positive_words:
            base = 1
        elif word in negative_words:
            base = -1

        if base != 0:
            neg = 0
            degree = 1
            for j in range(max(0, i - 3), i):
                if words[j] in negation_words:
                    neg += 1
                elif words[j] in degree_words:
                    degree *= degree_words[words[j]]
            if neg % 2 == 1:
                base *= -1
            base *= degree
        score += base
        i += 1

    if score > 0.5:
        return 1
    elif score < -0.5:
        return 0
    else:
        return 2  # 中性（暂不参与准确率计算）

# 4. 应用分析并比对真实标签
df['pred'] = df['review'].apply(analyze_sentiment)

# 只保留二分类样本用于验证准确率
filtered_df = df[df['pred'] != 2]
y_true = filtered_df['label']
y_pred = filtered_df['pred']

# 5. 输出准确率
acc = accuracy_score(y_true, y_pred)
print(f"情感分类准确率：{acc:.2%}")
print(filtered_df[['cat', 'label', 'pred', 'review']].head())

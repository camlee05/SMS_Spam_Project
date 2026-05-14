import pandas as pd
import matplotlib.pyplot as plt
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

df = pd.read_csv("data/spam.csv", encoding='latin-1')

# Giữ lại cột cần thiết
df = df[['v1', 'v2']]

# Đổi tên cột
df.columns = ['label', 'message']

# Xóa duplicate
df = df.drop_duplicates()

# Thống kê spam và ham
print(df['label'].value_counts())

# Vẽ biểu đồ
df['label'].value_counts().plot(kind='bar')

plt.title("Spam vs Ham")
plt.xlabel("Message Type")
plt.ylabel("Count")

plt.show()

# Tính % spam
spam_percent = (df['label'].value_counts()['spam'] / len(df)) * 100

print("Spam percentage:", spam_percent)

# Độ dài tin nhắn
df['message_length'] = df['message'].apply(len)

print(df.head())

# So sánh độ dài trung bình
print(df.groupby('label')['message_length'].mean())

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# test
print(transform_text("FREE entry!!! Win money now!!!"))

# preprocess toàn bộ dataset
df['processed_text'] = df['message'].apply(transform_text)

print(df[['message', 'processed_text']].head())
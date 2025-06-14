
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

nltk.download('stopwords')
arabic_stopwords = stopwords.words('arabic')

file_path = r"C:\Users\king\Downloads\AHD Arabic Healthcare Dataset\AHD Arabic Healthcare Dataset\AHD.xlsx"
df = pd.read_excel(file_path)


print(df.head())


def clean_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r'[إأآا]', 'ا', str(text))  
    text = re.sub(r'[ى]', 'ي', text)  
    text = re.sub(r'[ؤء]', 'ء', text)  
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  
    text = text.strip() 
    text = ' '.join([word for word in text.split() if word not in arabic_stopwords]) 
    return text


df['Cleaned_Question'] = df['Question'].apply(clean_text)


print(df[['Question', 'Cleaned_Question']].head())


from gensim.models import Word2Vec


sentences = df['Cleaned_Question'].apply(lambda x: x.split()).tolist()


model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
print("Word2Vec model trained successfully")


def get_vector(sentence):
    words = sentence.split()
    vector = np.zeros(100)
    count = 0
    for word in words:
        if word in model.wv:
            vector += model.wv[word]
            count += 1
    if count > 0:
        vector /= count
    return vector


X = np.array([get_vector(sentence) for sentence in df['Cleaned_Question']])


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Category'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

print(classification_report(
    y_test,
    y_pred,
    labels=label_encoder.transform(label_encoder.classes_),
    target_names=label_encoder.classes_,
    zero_division=0
))





def process_input(user_input):
    cleaned = clean_text(user_input)
    vector = get_vector(cleaned)
    return vector.reshape(1, -1)


print("مرحبًا بك في مساعد التشخيص الطبي! (اكتب 'خروج' لإنهاء المحادثة)")
while True:
    user_input = input("\nاكتب عرضك أو سؤالك: ")
    if user_input.strip().lower() == "خروج":
        print("شكرًا لاستخدامك المساعد. نتمنى لك الشفاء العاجل ")
        break
    
    vector = process_input(user_input)
    pred = clf.predict(vector)[0]
    category = label_encoder.inverse_transform([pred])[0]
 
    
    print(f"\nالتخصص الطبي المناسب: {category}")
 

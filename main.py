# Импортируем нужные библиотеки
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Загрузка данных
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Преобразование текста в числовые векторы с помощью TF-IDF
vectorizer = TfidfVectorizer(max_features=900000)
X_train_tfidf = vectorizer.fit_transform(train_data['text'])

# Разделение данных на обучающую и валидационную выборку
X_train, X_val, y_train, y_val = train_test_split(X_train_tfidf, train_data['sentiment'], test_size=0.2, random_state=42)

# Обучение модели SVM
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

# Получение предсказаний для валидационной выборки и вычисление f1-score
y_pred = svm_model.predict(X_val)
f1score = f1_score(y_val, y_pred, average='weighted')

# Выводим результат метрики f1-score
print("f1-score:", f1score)

# Преобразование текста из тестового набора в числовые векторы с помощью TF-IDF
X_test_tfidf = vectorizer.transform(test_data['text'])

# Предсказание эмоциональной категории текста в тестовых данных
sentiments = svm_model.predict(X_test_tfidf)

# Добавление предсказаний в новый столбец в тестовых данных
test_data['sentiment'] = sentiments

# Удаляем столбец 'text'
del test_data['text']

# Сохранение тестовых данных с предсказаниями в новый файл
test_data.to_csv('test_with_sentiment.csv', index=False)
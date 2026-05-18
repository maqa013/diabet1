import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Загрузка датасета
df = pd.read_csv("diabetes.csv")

# 2. Обработка пропущенных значений (0 физиологически невозможен для этих метрик)
# Заменяем нули медианным значением по каждому конкретному столбцу
columns_to_impute = ["Glucose", "BloodPressure",
                     "SkinThickness", "Insulin", "BMI"]
for col in columns_to_impute:
    df[col] = df[col].replace(0, df[col].median())

# 3. Разделение датасета на признаки (x) и целевую переменную (y)
x = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 4. Разделение данных на обучающую (80%) и тестовую (20%) выборки
# random_state фиксирует генератор случайных чисел для воспроизводимости результатов
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# 5. Инициализация и обучение модели логистической регрессии
# max_iter=1000 защищает от предупреждений о недостаточной сходимости (convergence warnings)
lr = LogisticRegression(max_iter=1000)
model = lr.fit(x_train, y_train)

# 6. Сохранение обученной модели в файл для последующего использования (деплоя)
pickle.dump(model, open("model.pkl", "wb"))

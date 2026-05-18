# diabet1
diabetes_data
# 🩺 Diabetes Prediction Web App (Logistic Regression)

Bu layihə, pasiyentlərin tibbi göstəricilərinə əsasən onlarda diabet riskinin olub-olmadığını proqnozlaşdıran bir Maşın Öyrənməsi (Machine Learning) tətbiqidir. Model **Logistic Regression** alqoritmi ilə qurulmuş və **Streamlit** kitabxanası vasitəsilə veb interfeysə köçürülmüşdür.

---

## 🚀 Layihənin Xüsusiyyətləri (Features)

* **Data Processing (Məlumatların Emalı):** Dataset daxilində tibbi olaraq `0` ola bilməyən dəyərlər (Qlükoza, Təzyiq, BMI, İnsulin və s.) analiz edilərək boşluqların yerinə median (orta) dəyərlər yerləşdirilmişdir (Imputation).
* **Model:** Scikit-learn kitabxanasının `LogisticRegression` alqoritmindən istifadə olunub.
* **Model Serialization:** Öyrədilmiş model `pickle` formatında yaddaşa yazılıb (`model.pkl`).
* **Web UI:** İstifadəçinin göstəriciləri daxil edib real vaxtda nəticəni və faiz etibarilə ehtimalı görə bilməsi üçün Streamlit interfeysi yaradılıb.

---

## 📂 Layihənin Strukturu (Project Structure)

```text
├── diabetes.csv          # Modelin öyrədilməsi üçün istifadə olunan dataset
├── train.py              # Modelin təmizlənməsi, öyrədilməsi və pkl formatında yazılması kodu
├── model.pkl             # Öyrədilmiş və yaddaşa saxlanılmış hazır model faylı
├── app.py                # Streamlit veb tətbiqinin interfeys kodu
└── README.md             # Layihə haqqında məlumat (Hazırda oxuduğunuz fayl)

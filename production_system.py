"""
PRODUCTION PHISHING DETECTOR - GERÇEK VERİ SETİ İLE KULLANIM
=============================================================

Adım 1: Veri setini indirin
Adım 2: CSV dosyasını proje klasörüne koyun
Adım 3: Aşağıdaki kodu kullanın
"""

import pandas as pd
import numpy as np
import pickle  # Modeli kaydetmek için
import os
from phishing_detector import PhishingDetector  # Ana kodunuzdaki class

class ProductionPhishingSystem:
    """Production ortamı için geliştirilmiş phishing tespit sistemi"""
    
    def __init__(self):
        self.detector = None
        self.model_path = "models/"
        self.data_path = "data/"
        
        # Klasörleri oluştur
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.data_path, exist_ok=True)
    
    def load_real_dataset(self, dataset_type='kaggle'):
        """
        Gerçek veri setini yükle
        
        Args:
            dataset_type: 'kaggle', 'local', veya 'custom'
        """
        
        if dataset_type == 'kaggle':
            # Kaggle'dan indirdiğiniz CSV dosyasını yükleyin
            # Örnek: https://www.kaggle.com/datasets/subhajournal/phishingemails
            
            try:
                # Veri setinin yapısına göre düzenleyin
                df = pd.read_csv('data/clean_emails.csv')  
                
                # Sütun isimlerini kontrol edin ve düzenleyin
                print("Mevcut sütunlar:", df.columns.tolist())
                
                # Genelde veri setlerinde şu sütunlar olur:
                # 'text' veya 'email' - Email içeriği
                # 'label' veya 'spam' - 0/1 veya ham/spam etiketi
                
                # Sütun isimlerini standartlaştır
                if 'email' in df.columns:
                    df.rename(columns={'email': 'text'}, inplace=True)
                if 'spam' in df.columns:
                    df.rename(columns={'spam': 'label'}, inplace=True)
                
                # Etiketleri 0/1'e dönüştür
                if df['label'].dtype == 'object':
                    df['label'] = df['label'].map({'ham': 0, 'spam': 1, 
                                                   'legitimate': 0, 'phishing': 1})
                
                print(f"\n✅ Veri seti yüklendi!")
                print(f"Toplam email sayısı: {len(df)}")
                print(f"Phishing email sayısı: {df['label'].sum()}")
                print(f"Normal email sayısı: {len(df) - df['label'].sum()}")
                
                return df
                
            except FileNotFoundError:
                print("❌ emails.csv dosyası bulunamadı!")
                print("Lütfen veri setini 'data/' klasörüne koyun.")
                return None
                
        elif dataset_type == 'local':
            # Kendi formatınızdaki veriyi yükleyin
            df = pd.read_csv('data/phishing_dataset.csv')
            
            # Veri yapınıza göre düzenlemeler yapın
            # Örnek:
            # df['label'] = df['is_phishing'].astype(int)
            # df['text'] = df['email_content']
            
            return df
            
        elif dataset_type == 'custom':
            # Farklı kaynaklardan veri toplama
            return self.create_custom_dataset()
    
    def create_custom_dataset(self):
        """
        Birden fazla kaynaktan veri birleştirme
        """
        datasets = []
        
        # 1. SpamAssassin verisi
        if os.path.exists('data/spam_assassin.csv'):
            df1 = pd.read_csv('data/spam_assassin.csv')
            datasets.append(df1)
        
        # 2. Enron verisi
        if os.path.exists('data/enron.csv'):
            df2 = pd.read_csv('data/enron.csv')
            datasets.append(df2)
        
        # 3. Kendi topladığınız veriler
        if os.path.exists('data/custom_phishing.csv'):
            df3 = pd.read_csv('data/custom_phishing.csv')
            datasets.append(df3)
        
        if datasets:
            # Tüm veri setlerini birleştir
            combined_df = pd.concat(datasets, ignore_index=True)
            print(f"✅ {len(datasets)} farklı veri seti birleştirildi.")
            return combined_df
        else:
            print("❌ Hiçbir veri seti bulunamadı!")
            return None
    
    def train_and_save_model(self, df):
        """
        Modeli eğit ve kaydet
        """
        print("\n" + "="*60)
        print("MODEL EĞİTİMİ BAŞLIYOR")
        print("="*60)
        
        # En iyi modeli seç (önceki testlere göre)
        best_model = self.find_best_model(df)
        
        # Final modeli eğit
        self.detector = PhishingDetector(model_type=best_model)
        X_train, X_test, y_train, y_test = self.detector.prepare_data(df)
        self.detector.train_model(X_train, y_train)
        
        # Performansı göster
        metrics = self.detector.evaluate_model(X_test, y_test)
        
        # Modeli kaydet
        self.save_model()
        
        return metrics
    
    def find_best_model(self, df):
        """
        Hangi modelin en iyi performans gösterdiğini bul
        """
        print("\n🔍 En iyi model aranıyor...")
        
        models = ['naive_bayes', 'logistic_regression', 'svm']
        best_score = 0
        best_model = 'naive_bayes'
        
        for model_type in models:
            print(f"\nTest ediliyor: {model_type}")
            detector = PhishingDetector(model_type=model_type)
            X_train, X_test, y_train, y_test = detector.prepare_data(df)
            detector.train_model(X_train, y_train)
            
            # F1 score'a göre en iyiyi seç
            y_pred = detector.model.predict(X_test)
            from sklearn.metrics import f1_score
            score = f1_score(y_test, y_pred)
            
            print(f"F1 Score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model = model_type
        
        print(f"\n🏆 En iyi model: {best_model} (F1: {best_score:.4f})")
        return best_model
    
    def save_model(self):
        """
        Eğitilmiş modeli ve vectorizer'ı kaydet
        """
        # Model dosyasını kaydet
        model_file = os.path.join(self.model_path, 'phishing_model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(self.detector.model, f)
        
        # Vectorizer'ı kaydet
        vectorizer_file = os.path.join(self.model_path, 'vectorizer.pkl')
        with open(vectorizer_file, 'wb') as f:
            pickle.dump(self.detector.vectorizer, f)
        
        print(f"\n✅ Model kaydedildi: {model_file}")
        print(f"✅ Vectorizer kaydedildi: {vectorizer_file}")
    
    def load_model(self):
        """
        Kaydedilmiş modeli yükle
        """
        try:
            # Detector'ı başlat
            self.detector = PhishingDetector()
            
            # Model'i yükle
            model_file = os.path.join(self.model_path, 'phishing_model.pkl')
            with open(model_file, 'rb') as f:
                self.detector.model = pickle.load(f)
            
            # Vectorizer'ı yükle
            vectorizer_file = os.path.join(self.model_path, 'vectorizer.pkl')
            with open(vectorizer_file, 'rb') as f:
                self.detector.vectorizer = pickle.load(f)
            
            print("✅ Model başarıyla yüklendi!")
            return True
            
        except FileNotFoundError:
            print("❌ Kaydedilmiş model bulunamadı. Önce modeli eğitin!")
            return False
    
    def predict_from_file(self, email_file):
        """
        Dosyadan email okuyup tahmin yap
        """
        with open(email_file, 'r', encoding='utf-8') as f:
            email_text = f.read()
        
        return self.detector.predict_email(email_text)
    
    def batch_prediction(self, csv_file):
        """
        Toplu email tahmini
        """
        df = pd.read_csv(csv_file)
        predictions = []
        
        for email in df['text']:
            result = self.detector.predict_email(email)
            predictions.append(result['prediction'])
        
        df['prediction'] = predictions
        df.to_csv('predictions_output.csv', index=False)
        print(f"✅ Tahminler 'predictions_output.csv' dosyasına kaydedildi.")
        
        return df

# ==========================================
# KULLANIM ÖRNEKLERİ
# ==========================================

def main_production():
    """
    Production sistemini çalıştır
    """
    system = ProductionPhishingSystem()
    
    print("="*60)
    print("PHISHING DETECTION SYSTEM - PRODUCTION")
    print("="*60)
    
    # Seçenekleri sun
    print("\nNe yapmak istersiniz?")
    print("1. Yeni model eğit")
    print("2. Mevcut modeli yükle ve kullan")
    print("3. Toplu tahmin yap")
    
    choice = input("\nSeçiminiz (1/2/3): ")
    
    if choice == '1':
        # Yeni model eğitme
        print("\nVeri seti tipini seçin:")
        print("1. Kaggle veri seti (emails.csv)")
        print("2. Lokal veri seti")
        print("3. Birleştirilmiş veri setleri")
        
        data_choice = input("\nSeçiminiz (1/2/3): ")
        dataset_type = ['kaggle', 'local', 'custom'][int(data_choice)-1]
        
        # Veri setini yükle
        df = system.load_real_dataset(dataset_type)
        
        if df is not None:
            # Model eğit ve kaydet
            metrics = system.train_and_save_model(df)
            
            print("\n✅ Model eğitimi tamamlandı!")
            print("Artık tahmin yapabilirsiniz.")
    
    elif choice == '2':
        # Mevcut modeli kullan
        if system.load_model():
            while True:
                print("\n" + "-"*60)
                email_text = input("\nEmail metnini girin (çıkmak için 'q'):\n> ")
                
                if email_text.lower() == 'q':
                    break
                
                result = system.detector.predict_email(email_text)
                
                print("\n" + "="*40)
                if result['prediction'] == 'PHISHING':
                    print(f"⚠️  UYARI: Bu email PHISHING olabilir!")
                else:
                    print(f"✅ Bu email güvenli görünüyor.")
                
                print(f"Güven skoru: %{result['confidence']:.2f}")
                print("="*40)
    
    elif choice == '3':
        # Toplu tahmin
        if system.load_model():
            csv_file = input("\nTahmin yapılacak CSV dosyasının adı: ")
            system.batch_prediction(csv_file)

# ==========================================
# DOCKER İÇİN DEPLOYMENT
# ==========================================

"""
Dockerfile örneği:

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
"""

# ==========================================
# WEB API İÇİN FLASK ENTEGRASYONU
# ==========================================

"""
from flask import Flask, request, jsonify

app = Flask(__name__)
system = ProductionPhishingSystem()
system.load_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    email_text = data.get('email_text', '')
    
    result = system.detector.predict_email(email_text)
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"""

if __name__ == "__main__":
    main_production()
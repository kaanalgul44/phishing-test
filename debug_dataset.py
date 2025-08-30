"""
VERİ SETİNİ KONTROL VE DÜZELTME SCRİPTİ
Bu scripti önce çalıştırarak veri setinizin yapısını anlayın
"""

import pandas as pd
import os

def analyze_dataset(file_path):
    """
    Veri setini analiz et ve sorunları tespit et
    """
    print("="*60)
    print("VERİ SETİ ANALİZİ")
    print("="*60)
    
    # CSV'yi yükle
    df = pd.read_csv(file_path)
    
    print(f"\n📊 Veri Seti Bilgileri:")
    print(f"Dosya: {file_path}")
    print(f"Satır sayısı: {len(df)}")
    print(f"Sütun sayısı: {len(df.columns)}")
    
    print(f"\n📋 Sütun İsimleri:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1}. '{col}'")
    
    print(f"\n🔍 İlk 3 Satır:")
    print(df.head(3))
    
    # Email Type sütununu kontrol et
    if 'Email Type' in df.columns:
        print(f"\n📧 Email Type Değerleri:")
        print(df['Email Type'].value_counts())
        print(f"\nBenzersiz değerler: {df['Email Type'].unique()}")
    
    # Email Text sütununu kontrol et
    if 'Email Text' in df.columns:
        print(f"\n📝 Email Text Örneği:")
        print(df['Email Text'].iloc[0][:200] + "...")
        
        # Boş değer kontrolü
        null_count = df['Email Text'].isna().sum()
        if null_count > 0:
            print(f"\n⚠️ UYARI: {null_count} adet boş email metni var!")
    
    return df

def fix_dataset(df):
    """
    Veri setini düzelt ve production'a hazırla
    """
    print("\n" + "="*60)
    print("VERİ SETİ DÜZELTİLİYOR")
    print("="*60)
    
    # Gereksiz sütunu sil
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        print("✅ Gereksiz index sütunu silindi")
    
    # Sütun isimlerini değiştir
    column_mapping = {
        'Email Text': 'text',
        'Email Type': 'label'
    }
    
    df = df.rename(columns=column_mapping)
    print("✅ Sütun isimleri düzeltildi: text, label")
    
    # Label değerlerini kontrol et ve düzelt
    print("\n🏷️ Etiketler düzeltiliyor...")
    
    # Mevcut unique değerleri göster
    if 'label' in df.columns:
        unique_labels = df['label'].unique()
        print(f"Mevcut etiketler: {unique_labels}")
        
        # String ise sayıya çevir
        if df['label'].dtype == 'object':
            # Önce küçük harfe çevir
            df['label'] = df['label'].str.strip().str.lower()
            
            # Olası tüm etiket kombinasyonları
            label_map = {
                # Phishing varyasyonları (1)
                'phishing email': 1,
                'phishing': 1,
                'phish': 1,
                'spam': 1,
                'scam': 1,
                'fraudulent': 1,
                'malicious': 1,
                '1': 1,
                
                # Legitimate varyasyonları (0)
                'safe email': 0,
                'legitimate email': 0,
                'legitimate': 0,
                'safe': 0,
                'ham': 0,
                'normal': 0,
                'genuine': 0,
                'valid': 0,
                '0': 0
            }
            
            # Etiketleri dönüştür
            df['label'] = df['label'].map(label_map)
            
            # NaN kontrolü
            nan_count = df['label'].isna().sum()
            if nan_count > 0:
                print(f"⚠️ {nan_count} adet tanınmayan etiket var!")
                print("Bu satırlar silinecek...")
                df = df.dropna(subset=['label'])
            
            # Integer'a çevir
            df['label'] = df['label'].astype(int)
            
        print(f"✅ Etiketler düzeltildi:")
        print(f"   - Phishing (1): {(df['label'] == 1).sum()} adet")
        print(f"   - Legitimate (0): {(df['label'] == 0).sum()} adet")
    
    # Boş text'leri temizle
    if 'text' in df.columns:
        before = len(df)
        df = df.dropna(subset=['text'])
        after = len(df)
        if before > after:
            print(f"✅ {before - after} adet boş email metni silindi")
    
    # Duplicate kontrolü
    before = len(df)
    df = df.drop_duplicates(subset=['text'])
    after = len(df)
    if before > after:
        print(f"✅ {before - after} adet tekrar eden email silindi")
    
    # Veri dengesini kontrol et
    print("\n📊 Veri Dengesi:")
    balance = df['label'].value_counts()
    phishing_ratio = balance[1] / len(df) * 100 if 1 in balance.index else 0
    print(f"   Phishing oranı: %{phishing_ratio:.1f}")
    
    if phishing_ratio < 20 or phishing_ratio > 80:
        print("   ⚠️ UYARI: Veri dengesiz! Daha iyi sonuç için dengeli veri kullanın.")
    else:
        print("   ✅ Veri dengesi uygun!")
    
    return df

def save_clean_dataset(df, output_path='data/clean_emails.csv'):
    """
    Temizlenmiş veriyi kaydet
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Temiz veri seti kaydedildi: {output_path}")
    print(f"   Toplam: {len(df)} email")
    return output_path

def main():
    """
    Ana fonksiyon
    """
    print("="*60)
    print("PHISHING VERİ SETİ DÜZELTME ARACI")
    print("="*60)
    
    # Veri setini bul
    possible_paths = [
        'data/emails.csv',
        'emails.csv',
        'data/phishing_emails.csv',
        'phishing_dataset.csv'
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    if not file_path:
        print("\n❌ CSV dosyası bulunamadı!")
        print("Lütfen CSV dosyanızı şu konumlardan birine koyun:")
        for path in possible_paths:
            print(f"  - {path}")
        
        # Manuel giriş
        manual_path = input("\nVeya dosya yolunu manuel girin: ")
        if os.path.exists(manual_path):
            file_path = manual_path
        else:
            print("❌ Dosya bulunamadı!")
            return
    
    # Analiz et
    df = analyze_dataset(file_path)
    
    # Düzeltmeyi sor
    print("\n" + "-"*60)
    choice = input("\nVeri setini düzeltmek ister misiniz? (e/h): ")
    
    if choice.lower() == 'e':
        # Düzelt
        df_clean = fix_dataset(df)
        
        # Kaydet
        output_path = save_clean_dataset(df_clean)
        
        print("\n" + "="*60)
        print("✅ İŞLEM TAMAMLANDI!")
        print("="*60)
        print("\nŞimdi production_system.py'yi çalıştırabilirsiniz.")
        print("Veri seti otomatik olarak 'data/clean_emails.csv' dosyasından yüklenecek.")
        
        # Production kodu için güncelleme önerisi
        print("\n📝 NOT: production_system.py dosyasında şu değişikliği yapın:")
        print("    df = pd.read_csv('data/emails.csv')")
        print("    yerine")
        print("    df = pd.read_csv('data/clean_emails.csv')")
        
if __name__ == "__main__":
    main()
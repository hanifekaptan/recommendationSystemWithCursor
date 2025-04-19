# Movie Recommender API

Bu proje, kullanıcıların film izleme geçmişlerine dayalı olarak basit bir film öneri sistemi API'si sunar. Kullanıcıları benzer zevklere sahip gruplara ayırmak için K-Means kümeleme algoritmasını kullanır ve bir kullanıcı için önerileri, aynı kümedeki diğer kullanıcıların beğendiği filmlere dayanarak oluşturur.

**Bu proje, AI destekli bir kod editörü olan [Cursor](https://cursor.sh/) kullanılarak geliştirilmiştir.**

## Özellikler

*   Sahte kullanıcı, film ve izleme aktivitesi verileri oluşturma (`recommendation_system.py`).
*   Verileri SQLite veritabanında saklama (SQLAlchemy kullanarak).
*   Kullanıcı izleme verilerine dayalı olarak K-Means modelini eğitme (`scikit-learn` kullanarak).
*   Eğitilmiş modeli, ölçekleyiciyi ve kullanıcı küme bilgilerini kaydetme (`joblib` kullanarak).
*   Kullanıcı kümelerini PCA ile görselleştirme (`matplotlib`/`seaborn` kullanarak).
*   Belirli bir kullanıcı için film önerileri sunan bir FastAPI API'si (`main.py`).

## Proje Yapısı

```
.
├── recommendation_system.py  # Veri üretme, model eğitimi ve kaydetme betiği
├── main.py                   # FastAPI uygulaması
├── requirements.txt          # Gerekli Python kütüphaneleri
├── recommendation_system.db  # Oluşturulan SQLite veritabanı (betik çalıştırıldıktan sonra)
├── kmeans_model.pkl          # Kaydedilmiş K-Means modeli (betik çalıştırıldıktan sonra)
├── scaler.pkl                # Kaydedilmiş StandardScaler (betik çalıştırıldıktan sonra)
├── user_clusters.pkl         # Kaydedilmiş kullanıcı-küme eşleşmesi (betik çalıştırıldıktan sonra)
└── user_clusters_pca.png     # Kullanıcı kümelerinin PCA görselleştirmesi (betik çalıştırıldıktan sonra)
└── README.md                 # Bu dosya
```

## Kurulum

1.  **Projeyi klonlayın veya indirin:**
    ```bash
    # Eğer git kullanıyorsanız:
    # git clone <repository_url>
    # cd <repository_directory>
    ```

2.  **Python Sanal Ortamı Oluşturun (Önerilir):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Bağımlılıkları Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

## Kullanım

1.  **Veri Oluşturma ve Model Eğitimi:**
    İlk olarak, veritabanını oluşturmak, verileri doldurmak, modeli eğitmek ve gerekli dosyaları kaydetmek için `recommendation_system.py` betiğini çalıştırın:
    ```bash
    python recommendation_system.py
    ```
    Bu işlem tamamlandığında, `.db` ve `.pkl` dosyaları ile `.png` görselleştirme dosyası proje dizininde oluşturulacaktır.

2.  **API Sunucusunu Başlatma:**
    FastAPI sunucusunu başlatmak için aşağıdaki komutlardan birini kullanın:
    ```bash
    python main.py
    ```
    veya (geliştirme sırasında otomatik yeniden yükleme için):
    ```bash
    uvicorn main:app --reload
    ```
    Sunucu varsayılan olarak `http://127.0.0.1:8000` adresinde çalışacaktır.

3.  **API'yi Kullanma:**
    *   **API Dokümantasyonu (Swagger UI):** Tarayıcınızda `http://127.0.0.1:8000/docs` adresine gidin. Buradan mevcut endpoint'leri görebilir ve interaktif olarak test edebilirsiniz.
    *   **Öneri Alma:** Belirli bir kullanıcı için öneri almak üzere `/recommend/{user_id}` endpoint'ine GET isteği gönderin. Örneğin, kullanıcı 1 için 5 öneri almak için:
        `http://127.0.0.1:8000/recommend/1?n_recommendations=5`
        `user_id`'yi (1 ile 300 arasında) ve isteğe bağlı olarak `n_recommendations` parametresini değiştirerek farklı kullanıcılar ve öneri sayıları için istek yapabilirsiniz.

## Notlar

*   Bu proje, temel bir öneri sistemi mantığını göstermek amacıyla oluşturulmuştur.
*   Veriler sahtedir ve rastgele üretilmiştir.
*   K-Means tabanlı kümeleme, kullanıcıları gruplamak için kullanılan birçok yöntemden sadece biridir. 
# 📊 Bilişsel Çarpıtma Tespit Sistemi - Deneysel Değerlendirme Sonuçları

Bu rapor `evaluate.py` betiği tarafından otomatik olarak üretilmiştir.

### 1. Aşama 1 (Gatekeeper) Değerlendirme Sonuçları

Aşama 1 modeli, kullanıcının girdiği metinde herhangi bir bilişsel çarpıtma olup olmadığını (VAR/YOK) tespit eden ikili sınıflandırıcıdır.

- **Toplam Test Örneği:** `448`
- **Genel Doğruluk (Accuracy):** `%97.32`
- **Macro F1-Score:** `%97.32`
- **Weighted F1-Score:** `%97.32`

| Sınıf / Kategori | Precision (%) | Recall (%) | F1-Score (%) | Destek (Örnek Sayısı) |
| :--- | :---: | :---: | :---: | :---: |
| **YOK (Sağlıklı/Nötr)** | %95.69 | %99.11 | %97.37 | 224 |
| **VAR (Bilişsel Çarpıtma)** | %99.07 | %95.54 | %97.27 | 224 |
| **Macro Ortalama** | **%97.38** | **%97.32** | **%97.32** | 448 |
| **Weighted Ortalama** | **%97.38** | **%97.32** | **%97.32** | 448 |

### 2. Aşama 2 (Specialist) Değerlendirme Sonuçları

Aşama 2 modeli, tespit edilen bilişsel çarpıtmanın hangi alt türe (Kişiselleştirme, Zihin Okuma, Olumluyu Görmezden Gelme, Abartma, Aşırı Genelleme) ait olduğunu teşhis eden 5 sınıflı uzman modeldir.

- **Toplam Test Örneği:** `706`
- **Genel Doğruluk (Accuracy):** `%96.88`
- **Macro F1-Score:** `%96.89`
- **Weighted F1-Score:** `%96.88`

| Sınıf / Kategori | Precision (%) | Recall (%) | F1-Score (%) | Destek (Örnek Sayısı) |
| :--- | :---: | :---: | :---: | :---: |
| **kişiselleştirme** | %97.08 | %95.00 | %96.03 | 140 |
| **zihin okuma** | %100.00 | %97.86 | %98.92 | 140 |
| **olumluyu görmezden gelme** | %94.33 | %98.52 | %96.38 | 135 |
| **abartma** | %96.60 | %99.30 | %97.93 | 143 |
| **aşırı genelleme** | %96.53 | %93.92 | %95.21 | 148 |
| **Macro Ortalama** | **%96.91** | **%96.92** | **%96.89** | 706 |
| **Weighted Ortalama** | **%96.92** | **%96.88** | **%96.88** | 706 |

### 3. Tümleşik Hiyerarşik Pipeline Sonuçları (Uçtan Uca)

İki aşamalı mimarinin gerçek hayat senaryosunda birlikte çalıştırılmasıyla elde edilen genel teşhis performansı.

- **Toplam Test Örneği:** `930`
- **Genel Doğruluk (Accuracy):** `%95.16`
- **Macro F1-Score:** `%95.21`
- **Weighted F1-Score:** `%95.14`

| Sınıf / Kategori | Precision (%) | Recall (%) | F1-Score (%) | Destek (Örnek Sayısı) |
| :--- | :---: | :---: | :---: | :---: |
| **Sağlıklı / Yok** | %90.61 | %99.11 | %94.67 | 224 |
| **kişiselleştirme** | %96.97 | %91.43 | %94.12 | 140 |
| **zihin okuma** | %100.00 | %97.14 | %98.55 | 140 |
| **olumluyu görmezden gelme** | %94.81 | %94.81 | %94.81 | 135 |
| **abartma** | %96.58 | %98.60 | %97.58 | 143 |
| **aşırı genelleme** | %95.59 | %87.84 | %91.55 | 148 |
| **Macro Ortalama** | **%95.76** | **%94.82** | **%95.21** | 930 |
| **Weighted Ortalama** | **%95.30** | **%95.16** | **%95.14** | 930 |


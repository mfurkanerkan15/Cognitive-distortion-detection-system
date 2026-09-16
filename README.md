# 🧠 Türkçe Bilişsel Çarpıtma Tespit ve Sınıflandırma Sistemi
### (Turkish Cognitive Distortion Detector & Classifier)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mfurkanerkan15)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

Bu proje, **Bilişsel Davranışçı Terapi (BDT / CBT)** ilkelerine dayalı olarak Türkçe metinlerdeki **bilişsel çarpıtmaları (otomatik olumsuz düşünce kalıplarını)** tespit etmek ve sınıflandırmak amacıyla geliştirilmiş **iki aşamalı hiyerarşik bir Doğal Dil İşleme (NLP)** sistemidir.

Proje, **BERTurk (`dbmdz/bert-base-turkish-cased`)** mimarisi üzerinde ince ayar (fine-tuning) yapılmış modellerle çalışır ve kullanıcı dostu bir **Streamlit** arayüzü sunar.

---

## 🚀 Hugging Face Model Havuzu

Geliştirilen modeller doğrudan Hugging Face Hub üzerinde açık erişimli olarak yayınlanmıştır:

| Aşama | Model Adı & Hub Bağlantısı | Görev | Mimari |
| :--- | :--- | :--- | :--- |
| **Aşama 1 (Gatekeeper)** | [🤗 `mfurkanerkan15/cognitive-distortion-detector-tr`](https://huggingface.co/mfurkanerkan15/cognitive-distortion-detector-tr) | **İkili Sınıflandırma:** Metinde bilişsel çarpıtma **VAR** / **YOK** tespiti | BERTurk Binary Classification |
| **Aşama 2 (Specialist)** | [🤗 `mfurkanerkan15/cognitive-distortion-classifier-tr`](https://huggingface.co/mfurkanerkan15/cognitive-distortion-classifier-tr) | **Çok Sınıflı Teşhis:** 5 temel çarpıtma türünün sınıflandırılması | BERTurk Multi-class + Focal Loss |

---

## 🏗️ Sistem Mimarisi & Çalışma Mantığı

Doğrudan çok sınıflı sınıflandırma yerine **iki aşamalı hiyerarşik (Gatekeeper + Specialist)** yaklaşım benimsenmiştir:

```mermaid
flowchart TD
    A([📝 Kullanıcı Metni / Düşünce İfadesi]) --> B[Aşama 1: Gatekeeper Model\n'cognitive-distortion-detector-tr']
    B --> C{Çarpıtma Var mı?\nThreshold: 0.20}
    C -- Hayır / Eşik Altı --> D([✅ Sağlıklı / Nötr Düşünce])
    C -- Evet / Eşik Üstü --> E[Aşama 2: Specialist Model\n'cognitive-distortion-classifier-tr']
    E --> F[Kişiselleştirme]
    E --> G[Zihin Okuma]
    E --> H[Olumluyu Görmezden Gelme]
    E --> I[Abartma / Felaketleştirme]
    E --> J[Aşırı Genelleme]
    F & G & H & I & J --> K([📊 Detaylı Teşhis & Güven Skoru Raporu])
```

1. **Aşama 1 (Filtre / Gatekeeper):** Cümlenin nötr/sağlıklı mı yoksa çarpıtma içerip içermediğini yüksek hassasiyetle (`threshold=0.20`) analiz eder.
2. **Aşama 2 (Uzman Teşhis / Specialist):** Çarpıtma tespit edilirse, metin ikinci modele yönlendirilerek spesifik çarpıtma kategorisi ve güven oranı belirlenir.

---

## 🧩 Hedeflenen Bilişsel Çarpıtma Kategorileri

| Kategori | Açıklama | Örnek İfade |
| :--- | :--- | :--- |
| **👤 Kişiselleştirme** | Doğrudan bağlantısı olmayan veya kontrol dışındaki olumsuz olaylardan kendini sorumlu tutma. | *"Toplantı kötü geçti, kesin benim yüzümden oldu."* |
| **🔮 Zihin Okuma** | Yeterli kanıt olmaksızın başkalarının kendisi hakkında olumsuz düşündüğünü varsayma. | *"Bana bakıp gülümsedi ama içinden benimle dalga geçtiğini biliyorum."* |
| **🙈 Olumluyu Görmezden Gelme** | Başarıları, övgüleri ve olumlu gelişmeleri şansa bağlayıp yok sayma. | *"Sınavdan 100 aldım ama sorular çok basitti, bir başarım yok."* |
| **💥 Abartma (Felaketleştirme)** | Küçük bir aksiliği veya hatayı telafisi imkansız bir felaket gibi algılama. | *"Bu sunumu yapamazsam tüm kariyerim tamamen bitecek."* |
| **🔁 Aşırı Genelleme** | Tek bir olumsuz deneyimden yola çıkarak genel ve mutlak kurallar çıkarma ("her zaman", "asla", "hiç kimse"). | *"İlk mülakattan elendim, ben asla hiçbir işte başarılı olamayacağım."* |

---

## 💻 Kurulum & Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/mfurkanerkan15/Cognitive-distortion-detection-system.git
cd Cognitive-distortion-detection-system
```

### 2. Sanal Ortam Oluşturun ve Aktif Edin
```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / MacOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin

Standart kurulum (CPU):
```bash
pip install -r requirements.txt
```

NVIDIA GPU (CUDA) hızlandırması ile kurulum:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Web Arayüzünü Başlatın
```bash
streamlit run app.py
```
Tarayıcınızda otomatik olarak `http://localhost:8501` adresi açılacaktır.

---

## 📁 Proje Dosya Yapısı

```plaintext
cognitive-distortion-detector/
├── app.py                   # Streamlit web uygulaması ve çıkarım (inference) pipeline'ı
├── train_model1.py          # Aşama 1 Gatekeeper modeli eğitim betiği
├── train_model2.py          # Aşama 2 Specialist modeli eğitim ve değerlendirme betiği
├── train_model2_focal.py    # Dengesiz veri setleri için Focal Loss destekli Aşama 2 eğitimi
├── label_map.json           # Sınıf etiketleri haritası (ID <-> Label)
├── requirements.txt         # Gerekli Python kütüphaneleri
├── LICENSE                  # MIT Lisans dosyası
└── README.md                # Proje dokümantasyonu
```

---

## 🔬 Model Eğitimi & İleri Teknikler

- **Önceden Eğitilmiş Temel Model:** `dbmdz/bert-base-turkish-cased`
- **Optimizasyon:** AdamW optimizasyon algoritması, `linear warmup` ile öğrenme oranı planlaması.
- **Focal Loss Entegrasyonu (`train_model2_focal.py`):** Sınıflar arası dengesizlik durumunda zor örneklere daha fazla odaklanmak amacıyla özelleştirilmiş `FocalLossTrainer` uygulanmıştır ($\gamma = 2.0$).
- **Değerlendirme Metrikleri:** Accuracy, Weighted/Binary Precision, Recall ve F1-Score.

---

## ⚠️ Yasal ve Etik Bildirim (Disclaimer)

> [!NOTE]
> Bu proje, **akademik araştırma ve Doğal Dil İşleme (NLP) teknolojilerinin psikolojik metin analizindeki uygulanabilirliğini göstermek amacıyla deneysel olarak geliştirilmiştir.**
> 
> Sistem tarafından üretilen sonuçlar **tıbbi bir teşhis, klinik değerlendirme veya profesyonel psikoterapi yerine geçmez.** Ruh sağlığı konularında lütfen uzman bir psikiyatrist veya klinik psikoloğa danışınız.

---

## 👨‍💻 Geliştirici & İletişim

**M. Furkan Erkan**

- **Hugging Face:** [@mfurkanerkan15](https://huggingface.co/mfurkanerkan15)
- **GitHub:** [@mfurkanerkan15](https://github.com/mfurkanerkan15)

Proje ile ilgili geri bildirim, katkı veya sorularınız için issue açabilir veya Hugging Face üzerinden iletişime geçebilirsiniz.

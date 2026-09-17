"""
=============================================================================
CBT Bilişsel Çarpıtma Tespit Sistemi - Model Değerlendirme Betiği (evaluate.py)
=============================================================================
Bu betik:
 1. Aşama 1 (Gatekeeper - İkili Sınıflandırma: VAR / YOK)
 2. Aşama 2 (Specialist - 5 Bilişsel Çarpıtma Türü)
 3. Hiyerarşik Pipeline (Aşama 1 + Aşama 2 Tümleşik Test)

modellerinin performansını somut metriklerle değerlendirir:
 - Accuracy (Genel Doğruluk)
 - Precision, Recall, F1-Score (Macro, Weighted ve Sınıf Bazında)
 - Confusion Matrix (Karışıklık Matrisi)
 - Destek (Support - Örnek Dağılımı)

Çıktılar:
 - Terminalde biçimlendirilmiş detaylı tablolar
 - README.md dosyasına doğrudan yapıştırılabilir 'evaluation_results.md' raporu
 - 'evaluation_results.json' sayısal sonuç dosyası
 - (Opsiyonel) PNG formatında Karışıklık Matrisi grafikleri

Kullanım:
  python evaluate.py
  python evaluate.py --model1_data model1_data.csv --model2_data model2_data.csv
  python evaluate.py --mode model1
  python evaluate.py --mode model2
  python evaluate.py --mode pipeline
  python evaluate.py --save_plots
=============================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# VARSAYILAN AYARLAR
# ==========================================
DEFAULT_MODEL1_PATH = "mfurkanerkan15/cognitive-distortion-detector-tr"
DEFAULT_MODEL2_PATH = "mfurkanerkan15/cognitive-distortion-classifier-tr"
DEFAULT_MODEL1_DATA = "model1_data.csv"
DEFAULT_MODEL2_DATA = "model2_data.csv"
DEFAULT_LABEL_MAP = "label_map.json"
DEFAULT_THRESHOLD = 0.20
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 128
DEFAULT_OUTPUT_MD = "evaluation_results.md"
DEFAULT_OUTPUT_JSON = "evaluation_results.json"

FALLBACK_ID2LABEL_MODEL2 = {
    0: "kişiselleştirme",
    1: "zihin okuma",
    2: "olumluyu görmezden gelme",
    3: "abartma",
    4: "aşırı genelleme"
}


# ==========================================
# VERİ KÜMESİ SINIFI
# ==========================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


# ==========================================
# YARDIMCI FONKSİYONLAR
# ==========================================
def read_csv_safe(file_path):
    """Farklı encoding formatlarını deneyerek CSV okur."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1254", "iso-8859-9"]
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise RuntimeError(f"'{file_path}' okunurken hata oluştu: {e}")
    raise ValueError(f"'{file_path}' desteklenen hiçbir encoding ile okunamadı.")


def load_label_map(label_map_path=DEFAULT_LABEL_MAP):
    """Etiket eşleme dosyasını okur."""
    if os.path.exists(label_map_path):
        try:
            with open(label_map_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                id2label = {int(k): v for k, v in data.get("id2label", {}).items()}
                label2id = data.get("label2id", {})
                return id2label, label2id
        except Exception as e:
            print(f"⚠️ {label_map_path} okunamadı ({e}), varsayılan etiket haritası kullanılacak.")
    
    id2label = FALLBACK_ID2LABEL_MODEL2
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def normalize_text_label_model1(label):
    """Model 1 etiketlerini (VAR/YOK veya 0/1) 0 veya 1 tamsayısına dönüştürür."""
    if isinstance(label, (int, np.integer)):
        return int(label)
    if isinstance(label, (float, np.floating)):
        return int(label)
    
    s = str(label).strip().upper()
    if s in ["1", "VAR", "TRUE", "POSITIVE", "DISTORTION", "ÇARPITMA"]:
        return 1
    elif s in ["0", "YOK", "FALSE", "NEGATIVE", "HEALTHY", "NEUTRAL", "SAĞLIKLI", "NÖTR"]:
        return 0
    else:
        try:
            return int(s)
        except ValueError:
            raise ValueError(f"Model 1 için tanınmayan etiket değeri: '{label}'")


def normalize_text_label_model2(label, label2id):
    """Model 2 etiketlerini (Metin veya ID) 0..N aralığında tamsayıya dönüştürür."""
    if isinstance(label, (int, np.integer)):
        return int(label)
    if isinstance(label, (float, np.floating)):
        return int(label)
    
    s = str(label).strip()
    try:
        val = int(s)
        return val
    except ValueError:
        pass
    
    # Metin eşleştirme (küçük harfe ve boşluklara göre toleranslı)
    cleaned = s.lower().replace("İ", "i").replace("I", "ı")
    for orig_name, idx in label2id.items():
        orig_clean = orig_name.lower().replace("İ", "i").replace("I", "ı")
        if cleaned == orig_clean:
            return idx
    
    # Alternatif Türkçe karakter normalizasyonları
    tr_map = {
        "kisisellestirme": 0, "kişiselleştirme": 0,
        "zihin okuma": 1, "zihin_okuma": 1,
        "olumluyu gormezden gelme": 2, "olumluyu görmezden gelme": 2,
        "abartma": 3, "felaketlestirme": 3, "felaketleştirme": 3,
        "asiri genelleme": 4, "aşırı genelleme": 4, "asirigenelleme": 4
    }
    if cleaned in tr_map:
        return tr_map[cleaned]
    
    raise ValueError(f"Model 2 için tanınmayan etiket değeri: '{label}'")


def resolve_model_path(preferred_path, fallback_path):
    """Yerel model dizini varsa onu, yoksa Hugging Face Hub adresini seçer."""
    if preferred_path and os.path.exists(preferred_path):
        print(f"📂 Yerel model bulundu ve yükleniyor: {preferred_path}")
        return preferred_path
    print(f"🌐 Yerel klasör bulunamadı, Hugging Face Hub üzerinden yükleniyor: {fallback_path}")
    return fallback_path


# ==========================================
# MODEL TAHMİN (INFERENCE) FONKSİYONU
# ==========================================
def predict_dataloader(model, dataloader, device):
    """Model ile veri kümesi üzerinde tahmin üretir."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ==========================================
# METRİK HESAPLAMA VE RAPORLAMA
# ==========================================
def calculate_metrics(y_true, y_pred, target_names):
    """Detaylı sınıflandırma metriklerini sözlük formatında döndürür."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    prec_per_class, rec_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(target_names))), zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))

    class_metrics = []
    for i, name in enumerate(target_names):
        class_metrics.append({
            "class_id": i,
            "class_name": name,
            "precision": float(prec_per_class[i]),
            "recall": float(rec_per_class[i]),
            "f1_score": float(f1_per_class[i]),
            "support": int(support_per_class[i])
        })

    return {
        "accuracy": float(acc),
        "macro_avg": {
            "precision": float(prec_macro),
            "recall": float(rec_macro),
            "f1_score": float(f1_macro)
        },
        "weighted_avg": {
            "precision": float(prec_weighted),
            "recall": float(rec_weighted),
            "f1_score": float(f1_weighted)
        },
        "per_class": class_metrics,
        "confusion_matrix": cm.tolist(),
        "total_samples": len(y_true)
    }


def print_metrics_table(title, metrics, class_names):
    """Terminal için metrik tablosu yazdırır."""
    print("\n" + "=" * 75)
    print(f"📊 {title}")
    print("=" * 75)
    print(f"Toplam Test Örneği: {metrics['total_samples']}")
    print(f"Genel Doğruluk (Accuracy) : %{metrics['accuracy']*100:.2f}")
    print(f"Macro F1-Score            : %{metrics['macro_avg']['f1_score']*100:.2f}")
    print(f"Weighted F1-Score         : %{metrics['weighted_avg']['f1_score']*100:.2f}")
    print("-" * 75)
    print(f"{'Sınıf Adı':<28} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Destek (N)':<10}")
    print("-" * 75)
    for cm in metrics["per_class"]:
        print(f"{cm['class_name']:<28} | %{cm['precision']*100:>8.2f} | %{cm['recall']*100:>8.2f} | %{cm['f1_score']*100:>8.2f} | {cm['support']:>10}")
    print("-" * 75)
    print(f"{'Macro Ortalaması':<28} | %{metrics['macro_avg']['precision']*100:>8.2f} | %{metrics['macro_avg']['recall']*100:>8.2f} | %{metrics['macro_avg']['f1_score']*100:>8.2f} | {metrics['total_samples']:>10}")
    print(f"{'Ağırlıklı Ortalama (Weighted)':<28} | %{metrics['weighted_avg']['precision']*100:>8.2f} | %{metrics['weighted_avg']['recall']*100:>8.2f} | %{metrics['weighted_avg']['f1_score']*100:>8.2f} | {metrics['total_samples']:>10}")
    print("=" * 75)
    
    print("\n🔢 Karışıklık Matrisi (Confusion Matrix):")
    cm_arr = np.array(metrics["confusion_matrix"])
    header = "          " + "".join([f"{name[:8]:>10}" for name in class_names])
    print(header)
    for i, row in enumerate(cm_arr):
        row_str = f"{class_names[i][:9]:<10}" + "".join([f"{val:>10}" for val in row])
        print(row_str)
    print()


def generate_markdown_section(title, metrics, description=""):
    """README.md için Markdown formatında tablo oluşturur."""
    md = f"### {title}\n\n"
    if description:
        md += f"{description}\n\n"
    
    md += f"- **Toplam Test Örneği:** `{metrics['total_samples']}`\n"
    md += f"- **Genel Doğruluk (Accuracy):** `%{metrics['accuracy']*100:.2f}`\n"
    md += f"- **Macro F1-Score:** `%{metrics['macro_avg']['f1_score']*100:.2f}`\n"
    md += f"- **Weighted F1-Score:** `%{metrics['weighted_avg']['f1_score']*100:.2f}`\n\n"

    md += "| Sınıf / Kategori | Precision (%) | Recall (%) | F1-Score (%) | Destek (Örnek Sayısı) |\n"
    md += "| :--- | :---: | :---: | :---: | :---: |\n"
    for cm in metrics["per_class"]:
        md += f"| **{cm['class_name']}** | %{cm['precision']*100:.2f} | %{cm['recall']*100:.2f} | %{cm['f1_score']*100:.2f} | {cm['support']} |\n"
    md += f"| **Macro Ortalama** | **%{metrics['macro_avg']['precision']*100:.2f}** | **%{metrics['macro_avg']['recall']*100:.2f}** | **%{metrics['macro_avg']['f1_score']*100:.2f}** | {metrics['total_samples']} |\n"
    md += f"| **Weighted Ortalama** | **%{metrics['weighted_avg']['precision']*100:.2f}** | **%{metrics['weighted_avg']['recall']*100:.2f}** | **%{metrics['weighted_avg']['f1_score']*100:.2f}** | {metrics['total_samples']} |\n\n"
    return md


def save_confusion_matrix_plot(cm, class_names, filename, title="Confusion Matrix"):
    """Karışıklık matrisini PNG olarak kaydeder."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 6), dpi=150)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True
        )
        plt.title(title, fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Tahmin Edilen Sınıf", fontsize=11, fontweight="bold")
        plt.ylabel("Gerçek Sınıf", fontsize=11, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"📈 Karışıklık matrisi grafiği kaydedildi: {filename}")
    except Exception as e:
        print(f"⚠️ Grafik kaydedilemedi (Matplotlib/Seaborn eksik olabilir): {e}")


# ==========================================
# 1. AŞAMA 1 DEĞERLENDİRME (GATEKEEPER)
# ==========================================
def evaluate_model1(model_path=None, data_path=DEFAULT_MODEL1_DATA, test_size=0.20, batch_size=DEFAULT_BATCH_SIZE, device="cpu", save_plot=False):
    print("\n" + "#" * 60)
    print("🚀 [1/3] AŞAMA 1 (GATEKEEPER - VAR/YOK) DEĞERLENDİRMESİ")
    print("#" * 60)

    if not os.path.exists(data_path):
        print(f"❌ '{data_path}' veri dosyası bulunamadı! Lütfen dosya yolunu kontrol edin.")
        return None

    df = read_csv_safe(data_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"'{data_path}' içerisinde 'text' ve 'label' sütunları bulunmalıdır.")

    df["label"] = df["label"].apply(normalize_text_label_model1)
    
    # Train / Test ayrımı (Eğitimdeki random_state=42 ile birebir aynı test kümesini üretir)
    _, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df["label"])
    print(f"✅ Test kümesi oluşturuldu: {len(test_df)} örnek (Toplam: {len(df)})")
    print(f"📊 Test etiket dağılımı: YOK (0): {(test_df['label'] == 0).sum()}, VAR (1): {(test_df['label'] == 1).sum()}")

    model_source = resolve_model_path(model_path or "./model1", DEFAULT_MODEL1_PATH)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source).to(device)

    dataset = TextDataset(test_df["text"], test_df["label"], tokenizer, max_length=DEFAULT_MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    y_true, y_pred, _ = predict_dataloader(model, dataloader, device)
    class_names = ["YOK (Sağlıklı/Nötr)", "VAR (Bilişsel Çarpıtma)"]

    metrics = calculate_metrics(y_true, y_pred, class_names)
    print_metrics_table("Aşama 1 (Gatekeeper) Performans Sonuçları", metrics, class_names)

    if save_plot:
        save_confusion_matrix_plot(
            np.array(metrics["confusion_matrix"]),
            ["YOK", "VAR"],
            "confusion_matrix_model1.png",
            "Aşama 1 (Gatekeeper) Karışıklık Matrisi"
        )

    return metrics


# ==========================================
# 2. AŞAMA 2 DEĞERLENDİRME (SPECIALIST)
# ==========================================
def evaluate_model2(model_path=None, data_path=DEFAULT_MODEL2_DATA, label_map_path=DEFAULT_LABEL_MAP, test_size=0.15, batch_size=DEFAULT_BATCH_SIZE, device="cpu", save_plot=False):
    print("\n" + "#" * 60)
    print("🚀 [2/3] AŞAMA 2 (SPECIALIST - 5 TÜR TEŞHİSİ) DEĞERLENDİRMESİ")
    print("#" * 60)

    if not os.path.exists(data_path):
        print(f"❌ '{data_path}' veri dosyası bulunamadı! Lütfen dosya yolunu kontrol edin.")
        return None

    id2label, label2id = load_label_map(label_map_path)
    df = read_csv_safe(data_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"'{data_path}' içerisinde 'text' ve 'label' sütunları bulunmalıdır.")

    df["label"] = df["label"].apply(lambda x: normalize_text_label_model2(x, label2id))
    
    # Train / Test ayrımı (Eğitimdeki random_state=42 ile birebir aynı test kümesi)
    _, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df["label"])
    print(f"✅ Test kümesi oluşturuldu: {len(test_df)} örnek (Toplam: {len(df)})")
    
    # Model yolu seçimi (Varsayılan olarak ./model2 veya ./model2_focal kontrol edilir)
    candidate_path = model_path
    if not candidate_path:
        if os.path.exists("./model2_focal"):
            candidate_path = "./model2_focal"
        elif os.path.exists("./model2"):
            candidate_path = "./model2"
        else:
            candidate_path = None

    model_source = resolve_model_path(candidate_path, DEFAULT_MODEL2_PATH)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(model_source).to(device)

    # Modeldeki id2label ile eşle
    if hasattr(model.config, "id2label") and model.config.id2label:
        model_id2label = {int(k): v for k, v in model.config.id2label.items()}
        class_names = [model_id2label[i] for i in range(len(model_id2label))]
    else:
        class_names = [id2label[i] for i in range(len(id2label))]

    dataset = TextDataset(test_df["text"], test_df["label"], tokenizer, max_length=DEFAULT_MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    y_true, y_pred, _ = predict_dataloader(model, dataloader, device)

    metrics = calculate_metrics(y_true, y_pred, class_names)
    print_metrics_table("Aşama 2 (Specialist) Performans Sonuçları", metrics, class_names)

    if save_plot:
        save_confusion_matrix_plot(
            np.array(metrics["confusion_matrix"]),
            class_names,
            "confusion_matrix_model2.png",
            "Aşama 2 (Specialist) Karışıklık Matrisi"
        )

    return metrics


# ==========================================
# 3. TÜMLEŞİK HİYERARŞİK PİPELİNE DEĞERLENDİRME
# ==========================================
def evaluate_pipeline(model1_path=None, model2_path=None, model1_data=DEFAULT_MODEL1_DATA, model2_data=DEFAULT_MODEL2_DATA, threshold=DEFAULT_THRESHOLD, batch_size=DEFAULT_BATCH_SIZE, device="cpu"):
    """
    İki aşamalı hiyerarşik pipeline'ı (Gatekeeper -> Specialist) uçtan uca test eder.
    Sınıflar: 0: Sağlıklı/Yok, 1: Kişiselleştirme, 2: Zihin Okuma, 3: Olumluyu Görmezden Gelme, 4: Abartma, 5: Aşırı Genelleme
    """
    print("\n" + "#" * 60)
    print("🚀 [3/3] TÜMLEŞİK HİYERARŞİK PİPELİNE DEĞERLENDİRMESİ")
    print(f"⚙️ Aşama 1 Eşik Değeri (Threshold): {threshold}")
    print("#" * 60)

    if not os.path.exists(model1_data) or not os.path.exists(model2_data):
        print("⚠️ Pipeline değerlendirmesi için hem model1_data.csv hem de model2_data.csv gereklidir.")
        return None

    # Model Yükleme
    m1_src = resolve_model_path(model1_path or "./model1", DEFAULT_MODEL1_PATH)
    m2_src = resolve_model_path(model2_path or "./model2", DEFAULT_MODEL2_PATH)
    
    tok1 = AutoTokenizer.from_pretrained(m1_src)
    mod1 = AutoModelForSequenceClassification.from_pretrained(m1_src).to(device)
    mod1.eval()

    tok2 = AutoTokenizer.from_pretrained(m2_src)
    mod2 = AutoModelForSequenceClassification.from_pretrained(m2_src).to(device)
    mod2.eval()

    # Verileri hazırla
    df1 = read_csv_safe(model1_data)
    df1["label"] = df1["label"].apply(normalize_text_label_model1)
    _, test_df1 = train_test_split(df1, test_size=0.20, random_state=42, stratify=df1["label"])
    
    # Sadece YOK (0) olan test örnekleri -> pipeline'da sınıf 0 (Sağlıklı)
    healthy_test = test_df1[test_df1["label"] == 0].copy()
    healthy_test["pipeline_label"] = 0

    id2label, label2id = load_label_map()
    df2 = read_csv_safe(model2_data)
    df2["label"] = df2["label"].apply(lambda x: normalize_text_label_model2(x, label2id))
    _, test_df2 = train_test_split(df2, test_size=0.15, random_state=42, stratify=df2["label"])
    
    # Model 2 sınıfları -> pipeline'da 1..5 aralığına kaydırılır
    distorted_test = test_df2.copy()
    distorted_test["pipeline_label"] = distorted_test["label"] + 1

    combined_test = pd.concat([healthy_test[["text", "pipeline_label"]], distorted_test[["text", "pipeline_label"]]], ignore_index=True)
    print(f"✅ Birleşik Pipeline Test Kümesi: {len(combined_test)} örnek (Sağlıklı: {len(healthy_test)}, Çarpıtmalı: {len(distorted_test)})")

    pipeline_class_names = ["Sağlıklı / Yok"] + [id2label[i] for i in range(5)]

    y_true_list = []
    y_pred_list = []

    print("🔄 Hiyerarşik çıkarım çalıştırılıyor...")
    for idx, row in combined_test.iterrows():
        text = str(row["text"])
        true_label = int(row["pipeline_label"])

        # Aşama 1: Gatekeeper
        inputs1 = tok1(text, return_tensors="pt", truncation=True, max_length=DEFAULT_MAX_LENGTH).to(device)
        with torch.no_grad():
            probs1 = F.softmax(mod1(**inputs1).logits, dim=-1)
        score_var = probs1[0][1].item()

        if score_var > threshold:
            # Aşama 2: Specialist
            inputs2 = tok2(text, return_tensors="pt", truncation=True, max_length=DEFAULT_MAX_LENGTH).to(device)
            with torch.no_grad():
                probs2 = F.softmax(mod2(**inputs2).logits, dim=-1)
            pred_id = torch.argmax(probs2).item()
            pipeline_pred = pred_id + 1
        else:
            pipeline_pred = 0

        y_true_list.append(true_label)
        y_pred_list.append(pipeline_pred)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    metrics = calculate_metrics(y_true, y_pred, pipeline_class_names)
    print_metrics_table("Tümleşik Hiyerarşik Pipeline Performans Sonuçları", metrics, pipeline_class_names)
    return metrics


# ==========================================
# ANA ÇALIŞTIRICI
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="CBT Bilişsel Çarpıtma Modelleri Performans Değerlendirme Aracı")
    parser.add_argument("--mode", choices=["all", "model1", "model2", "pipeline"], default="all", help="Değerlendirilecek mod (Varsayılan: all)")
    parser.add_argument("--model1_data", default=DEFAULT_MODEL1_DATA, help=f"Aşama 1 veri CSV dosyası (Varsayılan: {DEFAULT_MODEL1_DATA})")
    parser.add_argument("--model2_data", default=DEFAULT_MODEL2_DATA, help=f"Aşama 2 veri CSV dosyası (Varsayılan: {DEFAULT_MODEL2_DATA})")
    parser.add_argument("--model1_path", default=None, help="Aşama 1 model yolu veya HF Repo (Opsiyonel)")
    parser.add_argument("--model2_path", default=None, help="Aşama 2 model yolu veya HF Repo (Opsiyonel)")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch boyutu (Varsayılan: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"Aşama 1 Eşik değeri (Varsayılan: {DEFAULT_THRESHOLD})")
    parser.add_argument("--save_plots", action="store_true", help="Karışıklık matrisi grafiklerini PNG olarak kaydet")
    parser.add_argument("--output_md", default=DEFAULT_OUTPUT_MD, help=f"Markdown çıktı dosyası (Varsayılan: {DEFAULT_OUTPUT_MD})")
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON, help=f"JSON çıktı dosyası (Varsayılan: {DEFAULT_OUTPUT_JSON})")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 75)
    print("🧠 Bilişsel Çarpıtma Tespit Sistemi - Model Değerlendirme")
    print(f"💻 Çalışma Cihazı: {device.upper()}")
    print("=" * 75)

    all_results = {}
    markdown_content = "# 📊 Bilişsel Çarpıtma Tespit Sistemi - Deneysel Değerlendirme Sonuçları\n\n"
    markdown_content += "Bu rapor `evaluate.py` betiği tarafından otomatik olarak üretilmiştir.\n\n"

    # 1. Aşama 1
    if args.mode in ["all", "model1"]:
        if os.path.exists(args.model1_data):
            m1_results = evaluate_model1(
                model_path=args.model1_path,
                data_path=args.model1_data,
                batch_size=args.batch_size,
                device=device,
                save_plot=args.save_plots
            )
            if m1_results:
                all_results["model1"] = m1_results
                markdown_content += generate_markdown_section(
                    "1. Aşama 1 (Gatekeeper) Değerlendirme Sonuçları",
                    m1_results,
                    "Aşama 1 modeli, kullanıcının girdiği metinde herhangi bir bilişsel çarpıtma olup olmadığını (VAR/YOK) tespit eden ikili sınıflandırıcıdır."
                )
        else:
            print(f"ℹ️ '{args.model1_data}' bulunamadığı için Aşama 1 değerlendirmesi atlandı.")

    # 2. Aşama 2
    if args.mode in ["all", "model2"]:
        if os.path.exists(args.model2_data):
            m2_results = evaluate_model2(
                model_path=args.model2_path,
                data_path=args.model2_data,
                batch_size=args.batch_size,
                device=device,
                save_plot=args.save_plots
            )
            if m2_results:
                all_results["model2"] = m2_results
                markdown_content += generate_markdown_section(
                    "2. Aşama 2 (Specialist) Değerlendirme Sonuçları",
                    m2_results,
                    "Aşama 2 modeli, tespit edilen bilişsel çarpıtmanın hangi alt türe (Kişiselleştirme, Zihin Okuma, Olumluyu Görmezden Gelme, Abartma, Aşırı Genelleme) ait olduğunu teşhis eden 5 sınıflı uzman modeldir."
                )
        else:
            print(f"ℹ️ '{args.model2_data}' bulunamadığı için Aşama 2 değerlendirmesi atlandı.")

    # 3. Tümleşik Pipeline
    if args.mode in ["all", "pipeline"]:
        if os.path.exists(args.model1_data) and os.path.exists(args.model2_data):
            pipe_results = evaluate_pipeline(
                model1_path=args.model1_path,
                model2_path=args.model2_path,
                model1_data=args.model1_data,
                model2_data=args.model2_data,
                threshold=args.threshold,
                batch_size=args.batch_size,
                device=device
            )
            if pipe_results:
                all_results["pipeline"] = pipe_results
                markdown_content += generate_markdown_section(
                    "3. Tümleşik Hiyerarşik Pipeline Sonuçları (Uçtan Uca)",
                    pipe_results,
                    "İki aşamalı mimarinin gerçek hayat senaryosunda birlikte çalıştırılmasıyla elde edilen genel teşhis performansı."
                )

    if all_results:
        # Markdown raporu kaydet
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"\n📄 README için hazır Markdown raporu kaydedildi: {args.output_md}")

        # JSON çıktı kaydet
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"💾 Sayısal sonuçlar JSON olarak kaydedildi: {args.output_json}")
    else:
        print("\n⚠️ Hiçbir veri dosyası bulunamadığı için değerlendirme çalıştırılamadı.")
        print(f"👉 Lütfen '{args.model1_data}' ve/veya '{args.model2_data}' dosyalarını proje ana dizinine ekleyin veya argüman olarak belirtin:")
        print("   python evaluate.py --model1_data <dosya1.csv> --model2_data <dosya2.csv>")

    print("\n✅ İşlem tamamlandı!")


if __name__ == "__main__":
    main()

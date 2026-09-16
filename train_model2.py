import pandas as pd
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

# ================= AYARLAR =================

MODEL_ADI = "dbmdz/bert-base-turkish-cased"
VERI_DOSYASI = "model2_data.csv"
LABEL_MAP_DOSYASI = "label_map.json"
CIKIS_KLASORU = "./model2"

EPOCH_SAYISI = 6  

id2label = {
    0: "kişiselleştirme",          
    1: "zihin okuma",              
    2: "olumluyu görmezden gelme", 
    3: "abartma",                  
    4: "aşırı genelleme"           
}

label2id = {v: k for k, v in id2label.items()}
NUM_LABELS = len(id2label)

# JSON olarak kaydet
with open(LABEL_MAP_DOSYASI, "w", encoding="utf-8") as f:
    json.dump({"id2label": id2label, "label2id": label2id}, f, ensure_ascii=False)

print(f"Toplam Tür Sayısı: {NUM_LABELS}")

# ================= VERİ HAZIRLIĞI =================

try:
    df = pd.read_csv(VERI_DOSYASI, encoding='utf-8')
except:
    df = pd.read_csv(VERI_DOSYASI, encoding='latin-1')

train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])

dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'test': Dataset.from_pandas(test_df)
})

print("Tokenizer yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ADI)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_datasets = dataset.map(preprocess_function, batched=True)
cols_to_remove = [col for col in tokenized_datasets["train"].column_names if col not in ["input_ids", "attention_mask", "label"]]
tokenized_datasets = tokenized_datasets.remove_columns(cols_to_remove)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ================= MODEL KURULUMU =================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ADI, 
    num_labels=NUM_LABELS, 
    id2label=id2label, 
    label2id=label2id
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Eğitim Cihazı: {device}")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1}

# ================= EĞİTİM =================

training_args = TrainingArguments(
    output_dir=CIKIS_KLASORU,
    learning_rate=3e-5,  
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    fp16=True, 
    num_train_epochs=EPOCH_SAYISI,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    dataloader_num_workers=0, 
    logging_steps=100        
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nModel2 Eğitimi Başlıyor...")
trainer.train()

print(f"\nModel Kaydediliyor: {CIKIS_KLASORU}")
trainer.save_model(CIKIS_KLASORU)
tokenizer.save_pretrained(CIKIS_KLASORU)

# ================= SONUÇ ANALİZİ =================

print("\n--- DETAYLI PERFORMANS ANALİZİ ---")

predictions = trainer.predict(tokenized_datasets["test"])
preds = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

class_names = [id2label[i] for i in range(NUM_LABELS)]
print("\nSınıflandırma Raporu:")
print(classification_report(true_labels, preds, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(true_labels, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değer')
plt.title('Karışıklık Matrisi')
plt.show()
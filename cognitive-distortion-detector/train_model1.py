import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

# ==========================================
MODEL_ADI = "dbmdz/bert-base-turkish-cased" 

VERI_DOSYASI = "model1_data.csv" 
CIKIS_KLASORU = "./model1"

EPOCH_SAYISI = 3

# ==========================================
# 1. VERİYİ YÜKLEME

print(f"'{VERI_DOSYASI}' yükleniyor...")
try:
    df = pd.read_csv(VERI_DOSYASI, encoding='utf-8')
except:
    df = pd.read_csv(VERI_DOSYASI, encoding='latin-1')

print(f"Toplam Veri: {len(df)}")
print(f"Dağılım:\n{df['label'].value_counts()}")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'test': Dataset.from_pandas(test_df)
})

# ==========================================
# 2. TOKENIZATION

print("Tokenizer hazırlanıyor...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ADI)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Sütun temizliği
cols_to_remove = [col for col in tokenized_datasets["train"].column_names if col not in ["input_ids", "attention_mask", "label"]]
tokenized_datasets = tokenized_datasets.remove_columns(cols_to_remove)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ==========================================
# 3. MODEL KURULUMU

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ADI, 
    num_labels=2, 
    id2label={0: "YOK", 1: "VAR"}, 
    label2id={"YOK": 0, "VAR": 1}
)

# ==========================================
# 4. EĞİTİM AYARLARI

training_args = TrainingArguments(
    output_dir=CIKIS_KLASORU,
    learning_rate=2e-5,     
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=EPOCH_SAYISI,
    weight_decay=0.01,
    eval_strategy="epoch",  
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ==========================================
# 5. BAŞLAT

print("Model 1 Eğitimi Başlıyor...")
trainer.train()

print(f"\nModel Kaydediliyor: {CIKIS_KLASORU}")
trainer.save_model(CIKIS_KLASORU)
tokenizer.save_pretrained(CIKIS_KLASORU)

print("\n--- Test Seti Sonuçları ---")
metrics = trainer.evaluate()
print(metrics)
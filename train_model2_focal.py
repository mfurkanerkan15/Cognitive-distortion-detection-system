import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

# ================= AYARLAR =================
MODEL_ADI = "dbmdz/bert-base-turkish-cased" 
VERI_DOSYASI = "model2_data.csv" 
CIKIS_KLASORU = "./model2_focal"
EPOCH_SAYISI = 5  

# ================= 1. ÖZEL TRAINER (FOCAL LOSS) =================

class FocalLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
    
        gamma = 2.0
        
        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, labels)
        
        pt = torch.exp(-ce_loss)
        
        # Formül: (1 - pt)^gamma * ce_loss
        focal_loss = ((1 - pt) ** gamma * ce_loss).mean()
        
        return (focal_loss, outputs) if return_outputs else focal_loss

# ================= 2. VERİ HAZIRLIĞI =================

print(f"'{VERI_DOSYASI}' yükleniyor...")
try:
    df = pd.read_csv(VERI_DOSYASI)
except:
    df = pd.read_csv(VERI_DOSYASI, encoding="latin-1")

# Train/Test Ayrımı
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])

dataset = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'test': Dataset.from_pandas(test_df)
})

# ================= 3. MODEL VE TOKENIZER =================

id2label = {
    0: "kişiselleştirme",
    1: "zihin okuma",
    2: "olumluyu görmezden gelme",
    3: "abartma",
    4: "aşırı genelleme"
}
label2id = {v: k for k, v in id2label.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_ADI)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ADI, 
    num_labels=5,
    id2label=id2label,
    label2id=label2id
)

# ================= 4. EĞİTİM AYARLARI =================

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
    save_total_limit=2,
    logging_steps=50
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1}

trainer = FocalLossTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

print("\n FOCAL LOSS ile Eğitim Başlıyor (Zor Mod!)...")
trainer.train()

print(f"\n Model Kaydediliyor: {CIKIS_KLASORU}")
trainer.save_model(CIKIS_KLASORU)
tokenizer.save_pretrained(CIKIS_KLASORU)

# Sonuçları Göster
print("\n--- TEST SONUÇLARI ---")
preds = trainer.predict(tokenized_datasets["test"])
print(classification_report(preds.label_ids, np.argmax(preds.predictions, axis=1), target_names=list(id2label.values())))
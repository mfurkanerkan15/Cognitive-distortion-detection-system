import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time


st.set_page_config(
    page_title="CBT Cognitive Distortion Analyzer",
    page_icon="🧠",
    layout="centered"
)

# Hugging Face Hub model yolları
MODEL1_PATH = "mfurkanerkan15/cognitive-distortion-detector-tr"
MODEL2_PATH = "mfurkanerkan15/cognitive-distortion-classifier-tr"
THRESHOLD = 0.20

device = "cuda" if torch.cuda.is_available() else "cpu"

# MODELLERİ YÜKLE 
@st.cache_resource
def load_models():
    try:
        tok1 = AutoTokenizer.from_pretrained(MODEL1_PATH)
        mod1 = AutoModelForSequenceClassification.from_pretrained(MODEL1_PATH).to(device)
        mod1.eval()
        
        tok2 = AutoTokenizer.from_pretrained(MODEL2_PATH)
        mod2 = AutoModelForSequenceClassification.from_pretrained(MODEL2_PATH).to(device)
        mod2.eval()
        
        id2label = mod2.config.id2label
        return tok1, mod1, tok2, mod2, id2label
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        return None, None, None, None, None

with st.spinner('Yapay zeka modelleri Hugging Face üzerinden yükleniyor...'):
    tok1, mod1, tok2, mod2, id2label = load_models()

if mod1 is None:
    st.warning("Modeller yüklenemedi. Lütfen Hugging Face model adreslerini doğrulayın.")
    st.stop()


def normalize_label(label):
    if not isinstance(label, str): 
        return str(label)
    mapping = {
        "kisisellestirme": "Kişiselleştirme",
        "asiri genelleme": "Aşırı Genelleme",
        "olumluyu gormezden gelme": "Olumluyu Görmezden Gelme",
        "zihin okuma": "Zihin Okuma",
        "abartma": "Abartma",
        "yok": "Yok"
    }
    key = label.lower().strip()
    return mapping.get(key, label.title())

def analiz_et(text):
    # --- Aşama 1: Varlık/Yokluk Tespiti
    inputs1 = tok1(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        probs1 = F.softmax(mod1(**inputs1).logits, dim=-1)
    
    score_var = probs1[0][1].item()
    
    if score_var > THRESHOLD:
        # --- Aşama 2: Tür Teşhisi
        inputs2 = tok2(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            probs2 = F.softmax(mod2(**inputs2).logits, dim=-1)
        
        pred_id = torch.argmax(probs2).item()
        ham_etiket = id2label[pred_id]
        guven = probs2[0][pred_id].item()
        
        return {
            "durum": "VAR",
            "tur": normalize_label(ham_etiket),
            "guven": guven,
            "tespit_orani": score_var
        }
    else:
        return {
            "durum": "YOK",
            "tur": "Sağlıklı / Nötr",
            "guven": (1 - score_var),
            "tespit_orani": score_var
        }

# ARAYÜZ TASARIMI 
st.title("🧠 CBT Cognitive Distortion Analyzer")
st.markdown("**İki Aşamalı Hiyerarşik Türkçe Doğal Dil İşleme (NLP) Sistemi**")
st.caption("BERTurk mimarisi tabanlı bilişsel davranışçı terapi (CBT) düşünce analiz aracı.")

text_input = st.text_area(
    "Analiz edilecek düşünceyi girin:", 
    height=110, 
    placeholder="Örn: Sınavdan düşük aldım, geleceğim tamamen mahvoldu..."
)

col1, col2 = st.columns([1, 4])
with col1:
    analiz_butonu = st.button("Analiz Et", type="primary")

if analiz_butonu:
    if len(text_input.strip()) < 5:
        st.warning("Lütfen analiz için geçerli ve anlamlı bir cümle giriniz.")
    else:
        progress_bar = st.progress(0, text="Hiyerarşik modeller çalıştırılıyor...")
        for p in range(100):
            time.sleep(0.005)
            progress_bar.progress(p + 1)
        
        sonuc = analiz_et(text_input)
        progress_bar.empty()
        
        st.divider()
        
        if sonuc["durum"] == "VAR":
            c1, c2 = st.columns(2)
            with c1:
                st.error("🚨 Bilişsel Çarpıtma Tespit Edildi")
                st.metric(label="Teşhis Edilen Tür", value=sonuc["tur"])
            with c2:
                st.metric(label="Uzman Model Güveni", value=f"%{sonuc['guven']*100:.1f}")
                st.caption(f"Filtre Tespit Olasılığı: %{sonuc['tespit_orani']*100:.1f}")
            
            st.info(f"💡 **Model Yorumu:** İfadede **{sonuc['tur']}** bilişsel çarpıtma kalıbı baskın olarak tespit edildi.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.success("✅ Sağlıklı / Bilişsel Çarpıtma Bulunamadı")
                st.metric(label="Genel Durum", value="Nötr / Rasyonel")
            with c2:
                st.metric(label="Model Eminliği", value=f"%{sonuc['guven']*100:.1f}")
            st.caption("Model bu ifadede hedeflenen 5 temel bilişsel çarpıtma türüne rastlamadı.")

st.markdown("---")
st.caption("Developed by M. Furkan Erkan")
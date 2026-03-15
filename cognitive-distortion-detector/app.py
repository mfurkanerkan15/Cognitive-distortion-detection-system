import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

# ================= SAYFA AYARLARI =================
st.set_page_config(
    page_title="CBT Analiz Sistemi",
    page_icon="🧠",
    layout="centered"
)

# ================= AYARLAR =================
MODEL1_PATH = "./fineTuned_model1_final"      
MODEL2_PATH = "./fineTuned_M2_cleaned_yeni"  
THRESHOLD = 0.20

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= MODELLERİ YÜKLE (ÖNBELLEKLİ) =================
# Bu kısım modelleri sadece 1 kere yükler, her seferinde bekletmez.
@st.cache_resource
def load_models():
    try:
        # Model 1
        tok1 = AutoTokenizer.from_pretrained(MODEL1_PATH)
        mod1 = AutoModelForSequenceClassification.from_pretrained(MODEL1_PATH).to(device)
        mod1.eval()
        
        # Model 2
        tok2 = AutoTokenizer.from_pretrained(MODEL2_PATH)
        mod2 = AutoModelForSequenceClassification.from_pretrained(MODEL2_PATH).to(device)
        mod2.eval()
        
        id2label = mod2.config.id2label
        return tok1, mod1, tok2, mod2, id2label
    except Exception as e:
        return None, None, None, None, None

# Yükleme ekranı (Spinner)
with st.spinner('Yapay Zeka Modelleri Yükleniyor...'):
    tok1, mod1, tok2, mod2, id2label = load_models()

if mod1 is None:
    st.error("Modeller yüklenemedi! Klasör yollarını kontrol edin.")
    st.stop()

# ================= YARDIMCI FONKSİYONLAR =================
def normalize_label(label):
    if not isinstance(label, str): return str(label)
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
    # --- ADIM 1: Model 1 (Tespit) ---
    inputs1 = tok1(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        probs1 = F.softmax(mod1(**inputs1).logits, dim=-1)
    
    score_var = probs1[0][1].item() # Varlık ihtimali
    
    if score_var > THRESHOLD:
        # --- ADIM 2: Model 2 (Teşhis) ---
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

# ================= ARAYÜZ TASARIMI =================
st.title("CBT Proje")
st.markdown("**Yapay Zeka Destekli Bilişsel Çarpıtma Analizi**")
st.write("Aklınızdan geçen düşünceyi aşağıya yazın, yapay zeka analiz etsin.")

# Giriş Kutusu
text_input = st.text_area("Analiz edilecek cümle:", height=100, placeholder="Örn: Sınavdan düşük aldım, ben tam bir aptalım...")

col1, col2 = st.columns([1, 4])
with col1:
    analiz_butonu = st.button("Analiz Et", type="primary")

if analiz_butonu:
    if len(text_input) < 3:
        st.warning("Lütfen daha uzun bir cümle giriniz.")
    else:
        # İşlem barı (Görsellik)
        progress_text = "Düşünce analiz ediliyor..."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        
        # Analizi yap
        sonuc = analiz_et(text_input)
        my_bar.empty() # Barı kaldır
        
        st.divider()
        
        # Sonuçları Göster
        if sonuc["durum"] == "VAR":
            c1, c2 = st.columns(2)
            with c1:
                st.error("**Bilişsel Çarpıtma Tespit Edildi**")
                st.metric(label="Teşhis Edilen Tür", value=sonuc["tur"])
            
            with c2:
                st.metric(label="Model Güveni", value=f"%{sonuc['guven']*100:.1f}")
                st.write(f"Tespit Skoru: %{sonuc['tespit_orani']*100:.1f}")
            
            st.info(f" **Modelin Yorumu:** Bu cümlede **{sonuc['tur']}** özellikleri baskın görünüyor.")
            
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.success("**Düşünce Sağlıklı Görünüyor**")
                st.metric(label="Durum", value="Nötr / Sağlıklı")
            
            with c2:
                st.metric(label="Model Eminliği", value=f"%{sonuc['guven']*100:.1f}")
            
            st.write("Model bu ifadede herhangi bir bilişsel çarpıtma bulamadı.")

# Alt Bilgi
st.markdown("---")
st.caption("Bu sistem akademik çalışma kapsamında geliştirilmiştir. Tıbbi teşhis yerine geçmez.")
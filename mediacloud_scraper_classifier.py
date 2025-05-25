import os
import io
import re
import requests
import pandas as pd
import datetime as dt
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, urlunparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from mediacloud.api import SearchApi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F
import streamlit as st
from tqdm import tqdm
from deep_translator import GoogleTranslator

# === CONFIG ===
MODEL_PATH = "mike-83/longformer_policy_classifier"
TEXT_SIM_THRESHOLD = 0.85
search_api = SearchApi(st.secrets["MC_API_KEY"])

# === STATIC COLLECTION MAPPING ===
collections_df = pd.DataFrame({
    "country": [...],  # truncated for brevity
    "collection_id": [...]  # truncated for brevity
})

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_auth_token=st.secrets["HF_TOKEN"])
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, use_auth_token=st.secrets["HF_TOKEN"])
    model.to("cpu")
    return tokenizer, model

tokenizer, model = load_model()

# === SUMMARIZER ===
summarizer = pipeline(
    "summarization",
    model="Falconsai/text_summarization",
    tokenizer="Falconsai/text_summarization",
    device=0 if torch.cuda.is_available() else -1
)

# === UTILS ===
def translate_text(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

def generate_summary(text, max_tokens=130):
    try:
        summary = summarizer(text[:1024], max_new_tokens=max_tokens, min_length=30, do_sample=False)
        return summary[0]["summary_text"]
    except:
        return ""

def classify_binary(text, url=""):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1).squeeze().tolist()
    label = "policy_decision" if probs[1] > probs[0] else "news"
    return label, max(probs)

def extract_text_from_url(url):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, 'html.parser')
        for tag in soup(['script', 'style', 'footer', 'nav', 'aside', 'form']):
            tag.decompose()
        title_tag = soup.find(['title', 'h1'])
        title = title_tag.get_text(strip=True) if title_tag else ""
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        content = [p for p in paragraphs if len(p.split()) > 5][:15]
        return title + "\n\n" + "\n".join(content) if len(content) >= 2 else None
    except:
        return None

def drop_near_duplicates(df, text_col="text", url_col="url", text_threshold=TEXT_SIM_THRESHOLD):
    df = df.copy()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(df[text_col])
    to_drop = set()
    for i in tqdm(range(len(df)), desc="🔍 Checking duplicates", unit="doc"):
        if i in to_drop:
            continue
        for j in range(i + 1, len(df)):
            if j in to_drop:
                continue
            sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
            if sim > text_threshold:
                to_drop.add(j)
    return df.drop(index=list(to_drop))

# === STREAMLIT UI ===
st.title("Media Cloud Scraper + Classifier")

country = st.selectbox("Scegli il Paese:", collections_df['country'].sort_values().unique())
topic = st.text_input("Inserisci il tema della policy:", "Food waste")
start_date = st.date_input("Data inizio:", dt.date(2024, 1, 1))
end_date = st.date_input("Data fine:", dt.date(2024, 1, 10))

if st.button("Esegui ricerca e classificazione"):
    try:
        collection_id = int(collections_df.loc[collections_df['country'] == country, 'collection_id'].values[0])
        query = f"{country} {topic} Policy"
        stories, _ = search_api.story_list(
            query=query,
            start_date=start_date,
            end_date=end_date,
            collection_ids=[collection_id]
        )

        results = []
        for story in stories:
            url = story.get("url", "")
            text = extract_text_from_url(url)
            if not text:
                continue
            text = translate_text(text)
            label, confidence = classify_binary(text, url)
            summary = generate_summary(text)
            results.append({
                "country": country,
                "query": query,
                "url": url,
                "text": text,
                "label": label,
                "confidence": round(confidence, 3),
                "summary": summary
            })

        if not results:
            st.warning("⚠️ Nessun testo valido trovato tra gli URL estratti.")
        else:
            df_out = pd.DataFrame(results)
            df_out_clean = drop_near_duplicates(df_out, text_col="text", url_col="url", text_threshold=TEXT_SIM_THRESHOLD)

            output = io.BytesIO()
            df_out_clean.to_excel(output, index=False, engine="openpyxl")

            st.download_button(
                label="📥 Scarica risultati in Excel",
                data=output.getvalue(),
                file_name=f"mediacloud_classified_{country.lower()}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.dataframe(df_out_clean[["url", "label", "confidence", "summary"]])

    except Exception as e:
        st.error(f"❌ Errore durante l'elaborazione: {e}")

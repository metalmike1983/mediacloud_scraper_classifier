import os
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
# === STATIC COLLECTION MAPPING ===
collections_df = pd.DataFrame({
    "country": [
        "Afghanistan", "Albania", "Algeria", "Angola", "Argentina", "Armenia", "Australia", "Austria",
        "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize",
        "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria",
        "Burkina Faso", "Burundi", "Cambodia", "Cameroon", "Canada", "Cape Verde", "Central African Republic",
        "Chad", "Chile", "China", "Colombia", "Comoros", "Congo", "Costa Rica", "Croatia", "Cuba", "Cyprus",
        "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt",
        "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland",
        "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea",
        "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran",
        "Iraq", "Ireland", "Israel", "Italy", "Ivory Coast", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya",
        "Kiribati", "Kosovo", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia",
        "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives",
        "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova",
        "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal",
        "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Korea", "North Macedonia",
        "Norway", "Oman", "Pakistan", "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines",
        "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia",
        "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia",
        "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands",
        "Somalia", "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname",
        "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste",
        "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda",
        "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan",
        "Vanuatu", "Vatican City", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"
    ],
    "collection_id": [34442685, 34442687, 34442688, 34442689, 34442690, 34442691, 34442692, 34442693,
                       34442694, 34442695, 34442696, 34442697, 34442698, 34442699, 34442700, 34442701,
                       34442702, 34442703, 34442704, 34442705, 34442706, 34442707, 34442708, 34442709,
                       34442710, 34442711, 34442712, 34442713, 34442714, 34442715, 34442716, 34442717,
                       34442718, 34442719, 34442720, 34442721, 34442722, 34442723, 34442724, 34442725,
                       34442726, 34442727, 34442728, 34442729, 34442730, 34442731, 34442732, 34442733,
                       34442734, 34442735, 34442736, 34442737, 34442738, 34442739, 34442740, 34442741,
                       34442742, 34442743, 34442744, 34442745, 34442746, 34442747, 34442748, 34442749,
                       34442750, 34442751, 34442752, 34442753, 34442754, 34442755, 34442756, 34442757,
                       34442758, 34442759, 34442760, 34442761, 34442762, 34442763, 34442764, 34442765,
                       34442766, 34442767, 34442768, 34442769, 34442770, 34442771, 34442772, 34442773,
                       34442774, 34442775, 34442776, 34442777, 34442778, 34442779, 34442780, 34442781,
                       34442782, 34442783, 34442784, 34442785, 34442786, 34442787, 34442788, 34442789,
                       34442790, 34442791, 34442792, 34442793, 34442794, 34442795, 34442796, 34442797,
                       34442798, 34442799, 34442800, 34442801, 34442802, 34442803, 34442804, 34442805,
                       34442806, 34442807, 34442808, 34442809, 34442810, 34442811, 34442812, 34442813,
                       34442814, 34442815, 34442816, 34442817, 34442818, 34442819, 34442820, 34442821,
                       34442822, 34442823, 34442824, 34442825, 34442826, 34442827, 34442828, 34442829,
                       34442830, 34442831, 34442832, 34442833, 34442834, 34442835, 34442836, 34442837,
                       34442838, 34442839]
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
            if text:
                text = translate_text(text)
                label, confidence = classify_binary(text, url)
                summary = generate_summary(text)
                results.append({
                    "country": country,
                    "query": query,
                    "url": url,
                    "text": text if text else "",
                    "label": label if label else "",
                    "confidence": round(confidence, 3) if confidence else 0.0,
                    "summary": summary if summary else ""
                })

        df_out = pd.DataFrame(results)
        df_out_clean = drop_near_duplicates(df_out, text_col="text", url_col="url")

        st.download_button(
            label="📥 Scarica risultati in Excel",
            data=df_out_clean.to_excel(index=False, engine="openpyxl"),
            file_name=f"mediacloud_classified_{country.lower()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        st.dataframe(df_out_clean[["url", "label", "confidence", "summary"]])

    except Exception as e:
        st.error(f"❌ Errore durante l'elaborazione: {e}")

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
    "country": ["Afghanistan", "Albania", "Algeria", "American Samoa", "Andorra", "Angola", "Anguilla", "Antarctica", "Antigua and Barbuda", "Argentina", "Armenia", "Aruba", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bermuda", "Bhutan", "Bolivia. Plurinational State of", "Bonaire. Sint Eustatius and Saba", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei Darussalam", "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Cayman Islands", "Central African Republic", "Chad", "Chile", "China", "Taiwan. Province of China", "Christmas Island", "Cocos (Keeling) Islands", "Colombia", "Comoros", "Congo", "Congo. The Democratic Republic of the", "Cook Islands", "Costa Rica", "Croatia", "Cuba", "Curaçao", "Cyprus", "Czechia", "Côte d'Ivoire", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Ethiopia", "Falkland Islands (Malvinas)", "Faroe Islands", "Fiji", "Finland", "France", "French Guiana", "French Polynesia", "French Southern Territories", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Gibraltar", "Greece", "Greenland", "Grenada", "Guadeloupe", "Guam", "Guatemala", "Guernsey", "Equatorial Guinea", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Holy See (Vatican City State)", "Honduras", "Hong Kong", "Hungary", "Iceland", "India", "Indonesia", "Iran. Islamic Republic of", "Iraq", "Ireland", "Isle of Man", "Israel", "Italy", "Jamaica", "Japan", "Jersey", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Korea. Democratic People's Republic of", "Korea. Republic of", "Kosovo", "Kuwait", "Kyrgyzstan", "Lao People's Democratic Republic", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Macao", "Macedonia. Republic of", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Martinique", "Mauritania", "Mauritius", "Mayotte", "Mexico", "Micronesia. Federated States of", "Moldova. Republic of", "Monaco", "Mongolia", "Montenegro", "Montserrat", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "New Caledonia", "New Zealand", "Nicaragua", "Niger", "Nigeria", "Niue", "Norfolk Island", "Northern Mariana Islands", "Norway", "Oman", "Pakistan", "Palau", "Palestine. State of", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Pitcairn", "Poland", "Portugal", "Puerto Rico", "Qatar", "Romania", "Russian Federation", "Rwanda", "Réunion", "Saint Barthélemy", "Saint Helena", "Saint Kitts and Nevis", "Saint Lucia", "Saint Martin (French part)", "Saint Pierre and Miquelon", "Saint Vincent and the Grenadines", "American Samoa", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Sint Maarten (Dutch part)", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Georgia and the South Sandwich Islands", "South Sudan", "Spain", "Sudan", "Suriname", "Svalbard and Jan Mayen", "eSwatini", "Sweden", "Switzerland", "Syrian Arab Republic", "Taiwan. Province of China", "Tajikistan", "Tanzania. United Republic of", "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Turks and Caicos Islands", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela. Bolivarian Republic of", "Viet Nam", "Virgin Islands. British", "Virgin Islands. U.S.", "Wallis and Futuna", "Western Sahara", "Yemen", "Zambia", "Zimbabwe"
    ],
    "collection_id": [38376339,34412107,34412286,34412058,34412104,34412237,34412386,38376410,34412355,34412043,34412196,34412036,34412282,34412245,34412123,34412268,34412049,34412132,34412366,34412217,34412298,34412305,34412177,34412150,34412376,34412045,34412336,34412066,38379247,34412257,34412241,34412233,34412047,38379378,38379418,34411590,38379387,34411583,34412207,38379433,38379436,34412295,34412193,34412361,34412091,34412368,34412358,38379588,34412281,34412042,34412175,34412266,34412323,34412184,34412374,34412239,34412292,34412173,34412412,34412350,34412078,34412198,34412279,34412471,34412288,34412470,34412418,34412338,34412034,34412259,34412277,34412363,34412208,34412146,34412482,34412145,34412473,34412093,34412312,34412310,34412409,34412202,34412270,34412477,34412352,34412332,34412462,34412116,34412063,34412327,34412470,34412263,34412317,34412443,34412303,34412389,34412466,34412306,34412252,34412394,34412118,34412392,34412284,34412423,34412271,34412255,34412391,34412372,34412082,34412056,34412102,34412072,34412415,34412126,34412301,34412434,34412127,34412340,34412071,34412420,34412160,34412437,34412343,34412125,38380274,38380279,38380281,38379746,38380287,34412111,34412429,34412370,34412402,34412243,34412080,34412222,34412381,34412294,34412087,34412134,34412215,38380320,34412427,34412325,34412319,34412097,34412201,34412188,34412425,34412321,34412248,34412468,34412330,34412168,34412380,34412382,34412212,34412098,34412113,34412253,38376341,34412095,34412342,34412060,34412171,34412083,34412272,34412274,34412148,34412265,34412399,34412480,34412158,34412313,34412261,34412416,34412337,34412297,34412242,34412235,34412232,34412053,34412360,34412075,38380789,34412190,34412141,34411586,34412231,34412162,34412058,34412109,34412143,34411588,34412050,38380807,34412475,34412170,34412308,34412474,34412464,34412152,34412061,34412137,34412155,34412238,34412055,34412439,34412356,34412379,34412384,34412040,34412038,34412223,34411591,34412453,34412361,34412129,34412085,34412328,34412431,34412192,34412204,34412405,34412348,34412131,38381094,34412139,34412290,34412251,38381103,34412114,34412476,34412234,34412117,34412346,34412411,34412387,34412246,34412220,34412089,34412334,34412182,34412100,34412396,34412406
    ]
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

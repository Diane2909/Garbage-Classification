import streamlit as st
import pandas as pd
import os
import numpy as np
import tensorflow as tf
import plotly.express as px
from PIL import Image

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS_DIR = os.path.join(BASE_DIR, "data/stats")
INPUT_DIR = os.path.join(BASE_DIR, "data/input")
MODEL_PATH = os.path.join(BASE_DIR, "models/final_CNN.keras")
PRED_PARQUET = os.path.join(BASE_DIR, "data/prediction_data.parquet")  # Fichier des prédictions Spark

CLASSES = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic"]

# Couleurs du Canvas
CANVA_GREEN = "#1E7046"
CANVA_BEIGE = "#D6D1C1"

st.set_page_config(page_title="Garbage Analysis Dashboard", layout="wide")

# --- STYLE CSS (VERT SUR BEIGE) ---
st.markdown(f"""
    <style>
    .stApp {{ background-color: {CANVA_BEIGE}; }}
    header[data-testid="stHeader"] {{ background-color: {CANVA_GREEN} !important; }}

    /* Zone d'Upload */
    [data-testid="stFileUploader"] section {{
        background-color: {CANVA_GREEN} !important;
        border: 2px dashed #FFFFFF !important;
        border-radius: 10px !important;
    }}
    [data-testid="stFileUploader"] section div, [data-testid="stFileUploader"] section p, [data-testid="stFileUploader"] section span {{
        color: #FFFFFF !important;
    }}
    [data-testid="stFileUploader"] section .stButton > button {{
        background-color: #FFFFFF !important;
        color: {CANVA_GREEN} !important;
    }}

    /* Écriture Générale */
    .stMarkdown, p, label, .stText, h1, h2, h3, span {{
        color: {CANVA_GREEN} !important;
    }}

    /* Métriques et Tableaux */
    div[data-testid="stMetricValue"] {{
        background-color: #FFFFFF;
        color: {CANVA_GREEN} !important;
        border: 1px solid {CANVA_GREEN};
        border-radius: 10px;
        padding: 10px;
    }}
    /* Style pour le tableau de confiance */
    .stDataFrame, div[data-testid="stTable"] {{
        background-color: #FFFFFF;
        border-radius: 10px;
        border: 1px solid {CANVA_GREEN};
    }}
    </style>
    """, unsafe_allow_html=True)


# --- CHARGEMENT ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


model = load_model()


def load_pq(name):
    path = os.path.join(STATS_DIR, name)
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()


df_counts = load_pq("count_by_class.parquet")
# On charge les prédictions pour le describe de confiance
df_preds = pd.read_parquet(PRED_PARQUET) if os.path.exists(PRED_PARQUET) else pd.DataFrame()

# --- HEADER ---
st.title("♻️ Garbage Classification Dashboard")

# --- SECTION 1 : STATS & CAMEMBERT ---
st.header("📌 Statistiques du Dataset")
col_stats, col_pie = st.columns([3, 2])

with col_stats:
    c1, c2 = st.columns(2)
    with c1:
        # 1. CHANGEMENT : Valeur fixée à 10464 comme demandé
        st.metric("Total Images (Dataset)", 10464)
        st.metric("Nb Classes", 6)
    with c2:
        st.metric("Images Train", 7324)
        st.metric("Images Test", 3140)

with col_pie:
    df_split = pd.DataFrame({"Usage": ["Train", "Test"], "Valeur": [7324, 3140]})
    fig_pie = px.pie(df_split, values='Valeur', names='Usage', hole=0.4,
                     color_discrete_sequence=[CANVA_GREEN, "#7DBE6F"])
    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color=CANVA_GREEN,
                          margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_pie, use_container_width=True)

# --- SECTION 2 : REPARTITION (BARPLOT) ---
st.divider()
st.header("📊 Répartition par Catégorie")
if not df_counts.empty:
    label_col = "class" if "class" in df_counts.columns else "category"
    fig_raw = px.bar(df_counts, x=label_col, y="count", color=label_col, text="count",
                     title="Volume d'images par classe")
    fig_raw.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color=CANVA_GREEN, xaxis=dict(color=CANVA_GREEN), yaxis=dict(color=CANVA_GREEN))
    fig_raw.update_traces(textposition='outside', textfont_color=CANVA_GREEN)
    st.plotly_chart(fig_raw, use_container_width=True)

# --- SECTION 3 : ANALYSE DE LA CONFIANCE (NEW DIV) ---
st.divider()
st.header("🎯 Analyse de la Confiance (Modèle)")

if not df_preds.empty and 'confidence' in df_preds.columns:
    col_desc, col_info = st.columns([1, 2])  # On rééquilibre les colonnes

    with col_desc:
        st.write("**Statistiques (Spark describe) :**")

        # 1. On récupère le describe standard (vertical)
        stats_conf = df_preds['confidence'].describe().to_frame()

        # 2. On renomme la colonne pour que ce soit plus joli
        stats_conf.columns = ['Valeurs']

        # 3. On formate pour n'avoir que 2 chiffres après la virgule
        stats_formatted = stats_conf.style.format("{:.2f}")

        # 4. Affichage du tableau vertical (plus lisible)
        st.table(stats_formatted)

    with col_info:
        # On garde tes métriques à droite mais on les espace
        avg_conf = df_preds['confidence'].mean()
        max_conf = df_preds['confidence'].max()
        min_conf = df_preds['confidence'].min()

        m1, m2, m3 = st.columns(3)
        m1.metric("Confiance Moyenne", f"{avg_conf:.2%}")
        m2.metric("Confiance Max", f"{max_conf:.2%}")
        m3.metric("Confiance Min", f"{min_conf:.2%}")

        st.info(
            "💡 Ces statistiques permettent d'évaluer la précision du modèle sur les images envoyées. Une confiance moyenne > 50% est un bon début pour ce type de dataset.")
else:
    st.info("Aucune donnée de confiance disponible. Lancez le script Spark pour alimenter cette section.")

# --- SECTION 4 : UPLOAD ---
st.divider()
st.header("Upload un ou plusieurs fichier(s) pour traitement")
uploaded_files = st.file_uploader("", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    grid = st.columns(3)
    for i, file in enumerate(uploaded_files):
        with grid[i % 3]:
            # Sauvegarde dans input pour le prochain passage Spark
            if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
            with open(os.path.join(INPUT_DIR, file.name), "wb") as f:
                f.write(file.getbuffer())

            img = Image.open(file)
            st.image(img, use_container_width=True)
            if model:
                img_p = img.convert('L').resize((64, 64))
                img_arr = np.array(img_p).reshape(1, 64, 64, 1)
                preds = model.predict(img_arr, verbose=0)[0]
                idx = np.argmax(preds)
                st.write(f"**PRÉDICTION : {CLASSES[idx].upper()}** ({np.max(preds):.1%})")
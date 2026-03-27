import streamlit as st
import pandas as pd
import os
import numpy as np
import tensorflow as tf
import plotly.express as px
from PIL import Image

# --- CONFIGURATION DES CHEMINS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS_DIR = os.path.join(BASE_DIR, "data/stats")
INPUT_DIR = os.path.join(BASE_DIR, "data/input")
ARCHIVE_DIR = os.path.join(BASE_DIR, "data/archive")  # Dossier où Spark déplace les images traitées
MODEL_PATH = os.path.join(BASE_DIR, "models/final_CNN.keras")
PRED_PARQUET = os.path.join(BASE_DIR, "data/prediction_data.parquet")

CLASSES = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic"]
CANVA_GREEN = "#1E7046"
CANVA_BEIGE = "#D6D1C1"

st.set_page_config(page_title="Garbage Analysis Dashboard", layout="wide")

# --- STYLE CSS (VERT SOMBRE & BEIGE) ---
st.markdown(f"""
    <style>
    .stApp {{ background-color: {CANVA_BEIGE}; }}
    header[data-testid="stHeader"] {{ background-color: {CANVA_GREEN} !important; }}

    /* Zone d'Upload - Rectangle Vert */
    [data-testid="stFileUploader"] section {{
        background-color: {CANVA_GREEN} !important;
        border: 2px dashed #FFFFFF !important;
        border-radius: 10px !important;
    }}
    /* Texte blanc dans l'uploader pour la visibilité */
    [data-testid="stFileUploader"] section div, [data-testid="stFileUploader"] section p, [data-testid="stFileUploader"] section span {{
        color: #FFFFFF !important;
    }}
    [data-testid="stFileUploader"] section .stButton > button {{
        background-color: #FFFFFF !important;
        color: {CANVA_GREEN} !important;
    }}

    /* Toute l'écriture en Vert Sombre */
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
    .stTable {{ background-color: #FFFFFF; border-radius: 10px; }}
    </style>
    """, unsafe_allow_html=True)



@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


model = load_model()


def load_pq(name):
    path = os.path.join(STATS_DIR, name)
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()


df_counts = load_pq("count_by_class.parquet")
df_preds = pd.read_parquet(PRED_PARQUET) if os.path.exists(PRED_PARQUET) else pd.DataFrame()


st.title("♻️ Garbage Classification Dashboard")
st.header("📌 Statistiques du Dataset")
c_met, c_pie = st.columns([3, 2])

with c_met:
    m1, m2 = st.columns(2)
    m1.metric("Total Images (Dataset)", 10464)  # Valeur fixe demandée
    m1.metric("Nb Classes", 6)
    m2.metric("Images Train", 7324)
    m2.metric("Images Test", 3140)

with c_pie:
    df_split = pd.DataFrame({"Usage": ["Train", "Test"], "Valeur": [7324, 3140]})
    fig_pie = px.pie(df_split, values='Valeur', names='Usage', hole=0.4,
                     color_discrete_sequence=[CANVA_GREEN, "#7DBE6F"])
    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color=CANVA_GREEN, margin=dict(t=30, b=0))
    st.plotly_chart(fig_pie, use_container_width=True)


st.divider()
st.header("📊 Répartition par Catégorie")
if not df_counts.empty:
    label_col = "class" if "class" in df_counts.columns else "category"
    fig_raw = px.bar(df_counts, x=label_col, y="count", color=label_col, text="count")
    fig_raw.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=CANVA_GREEN)
    fig_raw.update_traces(textposition='outside', textfont_color=CANVA_GREEN)
    st.plotly_chart(fig_raw, use_container_width=True)


st.divider()
st.header("🎯 Analyse de la Confiance (Modèle)")

if not df_preds.empty and 'confidence' in df_preds.columns:
    col_t, col_m = st.columns([1, 2])  # 1/3 pour le tableau, 2/3 pour les metrics

    with col_t:
        st.write("**Statistiques complètes :**")
        stats = df_preds['confidence'].describe().to_frame()
        stats.columns = ['Valeurs']
        st.table(stats.style.format("{:.4f}"))

    with col_m:
        st.write("**Indicateurs clés :**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Moyenne", f"{df_preds['confidence'].mean():.2%}")
        m2.metric("Médiane (50%)", f"{df_preds['confidence'].median():.2%}")
        m3.metric("Écart-type (Std)", f"{df_preds['confidence'].std():.4f}")

        st.write("---")

        m4, m5 = st.columns(2)
        m4.metric("Minimum", f"{df_preds['confidence'].min():.2%}")
        m5.metric("Maximum", f"{df_preds['confidence'].max():.2%}")

        st.info("💡 L'écart-type (std) mesure la stabilité.")
else:
    st.info("Aucune donnée de confiance disponible. Lancez le script de prédiction Spark.")

st.write("---")
st.write("**Confiance moyenne par catégorie :**")


label_col = "class" if "class" in df_preds.columns else "category"
df_avg_conf = df_preds.groupby(label_col)['confidence'].mean().reset_index()
df_avg_conf = df_avg_conf.sort_values('confidence', ascending=False)


fig_avg = px.bar(
    df_avg_conf,
    x=label_col,
    y="confidence",
    color=label_col,
    text_auto='.2%',  # Affiche le pourcentage exact sur la barre
    title="Précision moyenne du modèle par classe",
    labels={label_col: "Catégorie", "confidence": "Confiance Moyenne"}
)


fig_avg.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color=CANVA_GREEN,
    xaxis=dict(showgrid=False, color=CANVA_GREEN),
    yaxis=dict(showgrid=True, gridcolor="#CCCCCC", color=CANVA_GREEN, tickformat='.0%'),
    margin=dict(t=50, b=50),
    showlegend=False  # On cache la légende car les noms sont déjà sur l'axe X
)


fig_avg.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)

st.plotly_chart(fig_avg, use_container_width=True)

st.divider()
st.header("📤 Inférence & Gestion des Flux")
uploaded_files = st.file_uploader("", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(3)
    for i, file in enumerate(uploaded_files):
        with cols[i % 3]:

            path_in_archive = os.path.join(ARCHIVE_DIR, file.name)


            if os.path.exists(path_in_archive):

                img = Image.open(file)
                st.image(img, use_container_width=True)
                if model:

                    img_p = img.convert('L').resize((64, 64))
                    img_arr = np.array(img_p).reshape(1, 64, 64, 1)


                    preds = model.predict(img_arr, verbose=0)[0]
                    idx = np.argmax(preds)
                    confidence = np.max(preds)


                    st.success(f"### Résultat : {CLASSES[idx].upper()} ({confidence:.2%}) (Déjà archivé)")
            else:

                if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
                with open(os.path.join(INPUT_DIR, file.name), "wb") as f:
                    f.write(file.getbuffer())

                st.image(Image.open(file), use_container_width=True)
                st.warning("⚠️ Sera prédit à la prochaine màj")
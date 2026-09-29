import json
import time
from fpdf import FPDF
from google import genai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

st.set_page_config(
    page_title="Malocher Scouting ⚒️", layout="wide", page_icon="⚒️"
)

st.title("Malocher Scouting ⚒️")
st.markdown(
    "Universelle, datengestützte Spielersuche & Recommender System powered by **Gemini & Cosine Similarity**"
)

# API Key sichern (aus Secrets oder direkt)
if "GEMINI_API_KEY" in st.secrets:
  GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

client = genai.Client(api_key=GEMINI_API_KEY)
FALLBACK_MODELS = [
    'gemini-3.8-flash',
    'gemini-3.5-flash',
    'gemini-flash-latest',
]

SKILL_MAP = {
    "pace": "Tempo",
    "shooting": "Schuss",
    "passing": "Passen",
    "dribbling": "Dribbling",
    "defending": "Defensive",
    "physic": "Physis",
}


# ---------------------------------------------------------
# Feature Engineering (Malocher-Index & ROI / Schnäppchen)
# ---------------------------------------------------------
@st.cache_data
def load_data():
  df = pd.read_csv("FC26_20250921.csv", low_memory=False)
  skill_columns = [
      "pace",
      "shooting",
      "passing",
      "dribbling",
      "defending",
      "physic",
  ]
  df[skill_columns] = df[skill_columns].fillna(df[skill_columns].mean())

  # 1. Feature Engineering: Malocher-Index (Physis 45%, Defensive 35%, Tempo 20%)
  df["malocher_index"] = np.round(
      (df["physic"] * 0.45 + df["defending"] * 0.35 + df["pace"] * 0.20), 1
  )

  # 2. Feature Engineering: Entwicklungspotenzial & ROI-Faktor
  df["potential_growth"] = df["potential"] - df["overall"]
  val_in_mio = np.maximum(df["value_eur"] / 1_000_000, 0.1)
  df["roi_score"] = np.round(df["potential_growth"] / val_in_mio, 2)

  return df, skill_columns


df, skill_columns = load_data()


# PDF-Generator Hilfsfunktion
def create_pdf_report(club_name, query, report_text, top_matches_df):
  pdf = FPDF()
  pdf.add_page()

  # Header
  pdf.set_font("Helvetica", "B", 18)
  pdf.cell(0, 10, f"MALOCHER SCOUTING - BERICHT", ln=True, align="C")
  pdf.set_font("Helvetica", "I", 12)
  pdf.cell(0, 8, f"Verein: {club_name}", ln=True, align="C")
  pdf.line(10, 30, 200, 30)
  pdf.ln(8)

  # Anfrage
  pdf.set_font("Helvetica", "B", 11)
  pdf.cell(0, 7, f"Anforderungsprofil: {query}", ln=True)
  pdf.ln(4)

  # KI-Bericht Text
  pdf.set_font("Helvetica", "B", 13)
  pdf.cell(0, 8, "Chef-Scout Analyse:", ln=True)
  pdf.set_font("Helvetica", "", 10)

  # UTF-8 Säuberung für FPDF
  clean_text = (
      report_text.replace("**", "")
      .replace("##", "")
      .encode("latin-1", "replace")
      .decode("latin-1")
  )
  pdf.multi_cell(0, 6, clean_text)

  return pdf.output()


# ---------------------------------------------------------
# Sidebar: Hybrid-Suche & Interaktive Regler (Human-in-the-Loop)
# ---------------------------------------------------------
st.sidebar.header("🎛️ Hybrid-Suche & Feinjustierung")
st.sidebar.markdown(
    "Hier kannst du die Kriterien manuell anpassen (*Human-in-the-Loop*):"
)

malocher_mode = st.sidebar.checkbox(
    "⚒️ Malocher-Fokus erzwingen (Hoher Malocher-Index)", value=False
)
schnaeppchen_mode = st.sidebar.checkbox(
    "💎 Nur Schnäppchen & Talente (Hoher ROI-Score)", value=False
)

override_max_value = st.sidebar.slider(
    "Maximaler Marktwert (€)",
    min_value=500_000,
    max_value=100_000_000,
    value=15_000_000,
    step=500_000,
)
override_max_age = st.sidebar.slider(
    "Maximales Alter", min_value=16, max_value=40, value=28
)
override_min_potential = st.sidebar.slider(
    "Mindest-Potenzial (POT)", min_value=60, max_value=95, value=75
)

# ---------------------------------------------------------
# Haupt-Eingabe
# ---------------------------------------------------------
user_prompt = st.text_input(
    "Welches Profil suchst du und für welchen Verein?",
    "Schlage mir einen spielstarken Innenverteidiger für Schalke 04 vor",
)

if st.button("🔍 Scouting-Analyse starten"):
  with st.spinner(
      "Malocher Scouting analysiert das Vereinsprofil und durchsucht die"
      " Datenbank..."
  ):

    extraction_prompt = f"""
        Du bist ein weltklasse Chef-Scout im Profifußball. Extrahiere die Parameter als JSON aus der Anfrage.
        
        WICHTIG - DYNAMISCHE VEREINSERKENNUNG & IMPLIZITE CLUB-PROFILE:
        Analysiere, welcher Verein in der Anfrage genannt wird. Falls keine konkreten Zahlen genannt werden:
        - Spitzenvereine (Bayern, BVB, Leipzig, Leverkusen): max_value_eur: 50000000, min_overall: 78, min_potential: 82.
        - Ambitionierte Bundesligisten (Frankfurt, Stuttgart, Wolfsburg, Gladbach, Freiburg): max_value_eur: 15000000, min_potential: 78.
        - Mittelfeld / 2. Liga / Traditionsvereine (Schalke, Köln, HSV, Hertha, Bochum, Mainz, St. Pauli): max_value_eur: 4000000, max_age: 24, min_potential: 75.

        POSITIONSMAPPING (Deutsch/Jargon -> EA FC Englisch):
        - Malocher / Abräumer / Sechser / ZDM / DM -> 'CDM'
        - Achter / ZM -> 'CM'
        - Zehner / Spielmacher / ZOM -> 'CAM'
        - Innenverteidiger / IV -> 'CB'
        - Außenverteidiger / LV / RV -> 'LB' bzw. 'RB'
        - Stürmer / MS / Knipser -> 'ST'
        - Flügelspieler / LA / RA -> 'LW' bzw. 'RW'

        Erlaubte JSON-Schlüssel:
        - club_name (String)
        - position (String, ENGLISCHE Kürzel z. B. 'ST', 'CM', 'CB', 'RW', 'LW', 'CAM', 'CDM', 'RB', 'LB' oder null)
        - max_age (Integer oder null)
        - max_value_eur (Integer in Euro oder null)
        - min_overall (Integer oder null)
        - min_potential (Integer oder null)
        - preferred_foot ('Left' oder 'Right' oder null)
        - similar_to_player (String, Spielername oder null)

        Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt ohne Markdown (kein ```json)!
        Anfrage: "{user_prompt}"
        """

    raw_text = None
    for m in FALLBACK_MODELS:
      try:
        res = client.models.generate_content(
            model=m, contents=extraction_prompt
        )
        raw_text = res.text
        break
      except Exception:
        continue

    if raw_text:
      clean_json = (
          raw_text.strip().replace("```json", "").replace("```", "").strip()
      )
      params = json.loads(clean_json)

      st.session_state["params"] = params
      st.session_state["user_prompt"] = user_prompt

if "params" in st.session_state:
  params = st.session_state["params"]
  user_prompt = st.session_state["user_prompt"]

  st.write("**Extrahierte Suchkriterien (Inkl. KI & Sidebar-Filter):**")
  st.json(params)

  # Datenbank-Filterung
  filtered_df = df.copy()

  # Berücksichtigung von Sidebar-Reglern
  max_val = min(
      params.get("max_value_eur") or 999_999_999, override_max_value
  )
  max_a = min(params.get("max_age") or 99, override_max_age)
  min_pot = max(params.get("min_potential") or 0, override_min_potential)

  filtered_df = filtered_df[
      (filtered_df["value_eur"] <= max_val)
      & (filtered_df["age"] <= max_a)
      & (filtered_df["potential"] >= min_pot)
  ]

  if params.get("min_overall"):
    filtered_df = filtered_df[filtered_df["overall"] >= params["min_overall"]]
  if params.get("preferred_foot"):
    filtered_df = filtered_df[
        filtered_df["preferred_foot"].str.contains(
            params["preferred_foot"], na=False, case=False
        )
    ]
  if params.get("position"):
    filtered_df = filtered_df[
        filtered_df["player_positions"].str.contains(
            params["position"], na=False, case=False
        )
    ]

  # Zusatz-Filter aus Sidebar
  if malocher_mode:
    filtered_df = filtered_df[
        filtered_df["malocher_index"] >= 70.0
    ].sort_values(by="malocher_index", ascending=False)
  if schnaeppchen_mode:
    filtered_df = filtered_df[filtered_df["roi_score"] >= 1.5].sort_values(
        by="roi_score", ascending=False
    )

  if filtered_df.empty:
    st.warning(
        "Keine Spieler gefunden, die alle Kriterien erfüllen! Locke die Filter"
        " in der Sidebar etwas auf."
    )
  else:
    similar_to = params.get("similar_to_player")
    target_name = None

    if similar_to:
      match = df[
          df["short_name"].str.contains(similar_to, case=False, na=False)
      ]
      if not match.empty:
        target_player = match.sort_values(by="overall", ascending=False).iloc[
            [0]
        ]
        target_name = target_player.iloc[0]["short_name"]

        scaler = MinMaxScaler()
        scaled_skills = scaler.fit_transform(filtered_df[skill_columns])
        scaled_target = scaler.transform(target_player[skill_columns])

        sims = cosine_similarity(scaled_skills, scaled_target).flatten()
        filtered_df["match_score_%"] = np.round(sims * 100, 1)
        results = filtered_df[
            filtered_df["short_name"] != target_name
        ].sort_values(by="match_score_%", ascending=False)
      else:
        filtered_df["match_score_%"] = "-"
        results = filtered_df.sort_values(
            by=["potential", "overall"], ascending=False
        )
    else:
      filtered_df["match_score_%"] = "-"
      results = filtered_df.sort_values(
          by=["potential", "overall"], ascending=False
      )

    top_matches = results[[
        "short_name",
        "age",
        "player_positions",
        "overall",
        "potential",
        "value_eur",
        "malocher_index",
        "roi_score",
        "match_score_%",
    ]].head(5)

    display_matches = top_matches.rename(columns={
        "short_name": "Name",
        "age": "Alter",
        "player_positions": "Positionen",
        "overall": "Gesamtstärke (OVR)",
        "potential": "Potenzial (POT)",
        "value_eur": "Marktwert (€)",
        "malocher_index": "⚒️ Malocher-Index",
        "roi_score": "💎 ROI-Faktor",
        "match_score_%": "Match-Score (%)",
    })

    detected_club = params.get("club_name", "Verein")

    col1, col2 = st.columns([1.2, 1])

    with col1:
      st.subheader(f"📋 Top 5 Treffer für {detected_club}")
      st.dataframe(display_matches, use_container_width=True)

    with col2:
      st.subheader("🥊 Direct Head-to-Head Spieler-Vergleich")

      player_list = top_matches["short_name"].tolist()
      p1_selected = st.selectbox("Spieler 1 auswählen:", player_list, index=0)
      p2_selected = st.selectbox(
          "Spieler 2 auswählen:",
          player_list,
          index=min(1, len(player_list) - 1),
      )

      if p1_selected and p2_selected:
        p1_data = df[df["short_name"] == p1_selected].iloc[0]
        p2_data = df[df["short_name"] == p2_selected].iloc[0]

        german_skill_labels = [
            SKILL_MAP[col].upper() for col in skill_columns
        ]
        categories = skill_columns

        v1 = p1_data[categories].values.tolist() + [
            p1_data[categories].values[0]
        ]
        v2 = p2_data[categories].values.tolist() + [
            p2_data[categories].values[0]
        ]
        angles = [
            n / float(len(categories)) * 2 * np.pi
            for n in range(len(categories))
        ] + [0]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        plt.xticks(angles[:-1], german_skill_labels)
        ax.plot(angles, v1, label=p1_selected, color="#004D98", linewidth=2)
        ax.plot(angles, v2, label=p2_selected, color="#E30613", linewidth=2)
        ax.fill(angles, v1, alpha=0.15, color="#004D98")
        ax.fill(angles, v2, alpha=0.15, color="#E30613")
        plt.legend(loc="lower right")
        st.pyplot(fig)

    # ---------------------------------------------------------
    # Scouting-Bericht & PDF-Download
    # ---------------------------------------------------------
    st.subheader(f"📝 Scouting-Bericht für {detected_club}")
    report_prompt = f"""
        Du bist Chef-Scout bei {detected_club}. 
        Anfrage des Managements: '{user_prompt}'. 
        
        Hier sind die datenbasierten Top-Kandidaten:
        {top_matches.to_string()}
        
        Schreibe einen professionellen, fundierten Scouting-Bericht direkt an die Vereinsführung.
        Gehe explizit auf den ⚒️ Malocher-Index (Physis/Einsatz) und den 💎 ROI-Faktor (Entwicklungspotenzial) ein.
        """
    rep_text = None
    for m in FALLBACK_MODELS:
      try:
        rep_text = client.models.generate_content(
            model=m, contents=report_prompt
        ).text
        break
      except Exception:
        continue

    if rep_text:
      st.markdown(rep_text)

      # PDF Export Button
      try:
        pdf_bytes = create_pdf_report(
            detected_club, user_prompt, rep_text, top_matches
        )
        st.download_button(
            label="📄 Scouting-Bericht als PDF herunterladen",
            data=bytes(pdf_bytes),
            file_name=f"Scouting_Bericht_{detected_club}.pdf",
            mime="application/pdf",
        )
      except Exception as e:
        st.info("PDF-Export steht bereit.")

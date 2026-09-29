import json
import re
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF
from fpdf.enums import WrapMode, XPos, YPos
from google import genai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Malocher Scouting ⚒️",
    layout="wide",
    page_icon="⚒️",
)

st.title("⚒️ Malocher Scouting ⚒️")
st.markdown(
    "Universelle, datengestützte Spielersuche & Recommender System powered by "
    "**Gemini, Cosine Similarity & Tactical Fit**"
)

# ---------------------------------------------------------
# API / Gemini
# ---------------------------------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error(
        "GEMINI_API_KEY fehlt. Bitte in Streamlit unter "
        "Settings → Secrets hinterlegen."
    )
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

FALLBACK_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
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
# Session State
# ---------------------------------------------------------
if "selected_prompt" not in st.session_state:
    st.session_state["selected_prompt"] = ""
if "last_processed" not in st.session_state:
    st.session_state["last_processed"] = ""
if "shortlist" not in st.session_state:
    st.session_state["shortlist"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ---------------------------------------------------------
# Daten & Feature Engineering
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

    required_columns = [
        "short_name",
        "age",
        "overall",
        "potential",
        "value_eur",
        "player_positions",
        *skill_columns,
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Folgende Spalten fehlen in FC26_20250921.csv: "
            + ", ".join(missing)
        )

    for col in ["age", "overall", "potential", "value_eur", *skill_columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in skill_columns:
        df[col] = df[col].fillna(df[col].mean())

    if "long_name" not in df.columns:
        df["long_name"] = df["short_name"]
    else:
        df["long_name"] = df["long_name"].fillna(df["short_name"])

    # Bestehende Kennzahlen
    df["malocher_index"] = np.round(
        (
            df["physic"] * 0.45
            + df["defending"] * 0.35
            + df["pace"] * 0.20
        ),
        1,
    )

    df["potential_growth"] = df["potential"] - df["overall"]
    val_in_mio = np.maximum(df["value_eur"] / 1_000_000, 0.1)
    df["roi_score"] = np.round(df["potential_growth"] / val_in_mio, 2)

    return df, skill_columns


try:
    df, skill_columns = load_data()
except Exception as e:
    st.error(f"Fehler beim Laden der Spielerdaten: {e}")
    st.stop()

# ---------------------------------------------------------
# KI-Helfer
# ---------------------------------------------------------
def gemini_generate(prompt):
    last_error = None
    for model in FALLBACK_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            if getattr(response, "text", None):
                return response.text, None
        except Exception as exc:
            last_error = str(exc)
    return None, last_error


# ---------------------------------------------------------
# Tactical Fit
# ---------------------------------------------------------
TACTICAL_PROFILES = {
    "balanced": {
        "pace": 0.15,
        "shooting": 0.15,
        "passing": 0.20,
        "dribbling": 0.15,
        "defending": 0.20,
        "physic": 0.15,
    },
    "possession": {
        "pace": 0.10,
        "shooting": 0.10,
        "passing": 0.30,
        "dribbling": 0.20,
        "defending": 0.15,
        "physic": 0.15,
    },
    "counter": {
        "pace": 0.30,
        "shooting": 0.15,
        "passing": 0.20,
        "dribbling": 0.20,
        "defending": 0.05,
        "physic": 0.10,
    },
    "pressing": {
        "pace": 0.20,
        "shooting": 0.05,
        "passing": 0.10,
        "dribbling": 0.10,
        "defending": 0.25,
        "physic": 0.30,
    },
    "defensive": {
        "pace": 0.15,
        "shooting": 0.05,
        "passing": 0.15,
        "dribbling": 0.05,
        "defending": 0.35,
        "physic": 0.25,
    },
    "attacking": {
        "pace": 0.20,
        "shooting": 0.30,
        "passing": 0.15,
        "dribbling": 0.25,
        "defending": 0.05,
        "physic": 0.05,
    },
}

TACTICAL_KEYWORDS = {
    "possession": [
        "ballbesitz",
        "ballbesitzfußball",
        "kombinationsfußball",
        "spielaufbau",
        "spieleröffn",
        "technisch",
        "passstark",
    ],
    "counter": [
        "konter",
        "umschalt",
        "gegenstoß",
        "transition",
        "schnelle angriffe",
    ],
    "pressing": [
        "pressing",
        "gegenpressing",
        "aggressiv",
        "intensiv",
        "zweikampfstark",
        "arbeitsintensiv",
    ],
    "defensive": [
        "defensiv",
        "defensive stabilität",
        "abwehr",
        "kompakt",
        "defensivstark",
    ],
    "attacking": [
        "offensiv",
        "angriff",
        "torgefährlich",
        "torjäger",
        "kreativ",
        "abschlussstark",
    ],
}


def get_tactical_weights(prompt):
    prompt_lower = prompt.lower()
    active_profiles = []

    for profile, keywords in TACTICAL_KEYWORDS.items():
        if any(keyword in prompt_lower for keyword in keywords):
            active_profiles.append(profile)

    if not active_profiles:
        return TACTICAL_PROFILES["balanced"], ["Ausgewogen"]

    weights = {skill: 0.0 for skill in skill_columns}
    for profile in active_profiles:
        for skill, value in TACTICAL_PROFILES[profile].items():
            weights[skill] += value

    divisor = float(len(active_profiles))
    weights = {skill: value / divisor for skill, value in weights.items()}

    labels = {
        "possession": "Ballbesitz",
        "counter": "Umschalten/Konter",
        "pressing": "Pressing",
        "defensive": "Defensive",
        "attacking": "Offensive",
    }
    return weights, [labels[p] for p in active_profiles]


def calculate_tactical_fit(dataframe, weights):
    weighted_sum = np.zeros(len(dataframe))
    total_weight = sum(weights.values())
    for skill, weight in weights.items():
        weighted_sum += dataframe[skill].fillna(0).to_numpy() * weight
    return np.round(weighted_sum / total_weight, 1)


# ---------------------------------------------------------
# PDF
# ---------------------------------------------------------
def _pdf_safe_text(value):
    """Sanitize AI/user text for FPDF's built-in Helvetica font."""
    if value is None:
        return ""

    text = str(value)
    text = (
        text.replace("**", "")
        .replace("__", "")
        .replace("###", "")
        .replace("##", "")
        .replace("# ", "")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2011", "-")
        .replace("\u2212", "-")
        .replace("\u2022", "-")
        .replace("\u00a0", " ")
    )

    text = text.encode("latin-1", "replace").decode("latin-1")

    safe_lines = []
    for line in text.splitlines() or [""]:
        if not line:
            safe_lines.append("")
            continue

        parts = re.split(r"(\s+)", line)
        rebuilt = ""
        for part in parts:
            if part.isspace():
                rebuilt += part
            elif len(part) > 60:
                chunks = [part[i:i + 60] for i in range(0, len(part), 60)]
                rebuilt += "\n".join(chunks)
            else:
                rebuilt += part
        safe_lines.append(rebuilt)

    return "\n".join(safe_lines)


def _pdf_multiline(pdf, text, height=5, size=9):
    pdf.set_font("Helvetica", "", size)
    pdf.multi_cell(
        0,
        height,
        _pdf_safe_text(text),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )


def create_pdf_report(club_name, query, report_text, top_matches_df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 10, 10)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "MALOCHER SCOUTING - BERICHT", ln=True, align="C")

    pdf.set_font("Helvetica", "I", 12)
    safe_club = _pdf_safe_text(f"Verein: {club_name}")
    if len(safe_club) > 100:
        safe_club = safe_club[:97] + "..."
    pdf.cell(0, 8, safe_club.replace("\n", " "), ln=True, align="C")

    pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
    pdf.ln(8)

    pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(
        0,
        7,
        _pdf_safe_text(f"Anforderungsprofil: {query}"),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(
        0,
        8,
        "Chef-Scout Analyse:",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )

    _pdf_multiline(pdf, report_text, height=6, size=10)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(
        0,
        7,
        "Top-Kandidaten:",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )

    for _, row in top_matches_df.iterrows():
        line = (
            f"{row['long_name']} | OVR {row['overall']} | POT {row['potential']} | "
            f"Tactical Fit {row['tactical_fit']} | Malocher {row['malocher_index']} | "
            f"ROI {row['roi_score']}"
        )
        _pdf_multiline(pdf, line, height=5, size=9)

    output = pdf.output(dest="S")
    return output.encode("latin-1") if isinstance(output, str) else bytes(output)


# ---------------------------------------------------------
# Sidebar (Regler auf Maximalwerte gesetzt)
# ---------------------------------------------------------
st.sidebar.header("🎛️ Hybrid-Suche & Feinjustierung")
st.sidebar.markdown(
    "Hier kannst du die Kriterien manuell anpassen (*Human-in-the-Loop*):"
)

malocher_mode = st.sidebar.checkbox(
    "⚒️ Malocher-Fokus erzwingen (Index ≥ 70)", 
    value=False,
    help="Filtert ausschließlich Spieler mit einem Malocher-Index von 70 oder höher."
)
schnaeppchen_mode = st.sidebar.checkbox(
    "💎 Nur Schnäppchen & Talente (ROI ≥ 1,5)", 
    value=False,
    help="Filtert ausschließlich Spieler mit einem herausragenden ROI-Score von 1,5 oder höher."
)

override_max_value = st.sidebar.slider(
    "Maximaler Marktwert (€)",
    min_value=500_000,
    max_value=100_000_000,
    value=100_000_000,
    step=500_000,
    help="Obergrenze für den aktuellen Marktwert der Spieler."
)
override_max_age = st.sidebar.slider(
    "Maximales Alter",
    min_value=16,
    max_value=40,
    value=40,
    help="Höchstalter der zu berücksichtigenden Spieler."
)
override_min_potential = st.sidebar.slider(
    "Mindest-Potenzial (POT)",
    min_value=60,
    max_value=95,
    value=95,
    help="Mindestwert für das erwartete Maximalpotenzial (POT)."
)

st.sidebar.divider()
st.sidebar.subheader("⭐ Meine Shortlist")

if st.session_state["shortlist"]:
    for player_name in list(st.session_state["shortlist"]):
        s_col1, s_col2 = st.sidebar.columns([4, 1])
        s_col1.write(player_name)
        if s_col2.button("×", key=f"remove_{player_name}"):
            st.session_state["shortlist"].remove(player_name)
            st.rerun()
else:
    st.sidebar.caption("Noch keine Spieler gespeichert.")

# ---------------------------------------------------------
# Search Input
# ---------------------------------------------------------
st.markdown("### 🔎 Spielerprofil definieren")

with st.form(key="search_form"):
    user_prompt = st.text_input(
        "Suchanfrage",
        value=st.session_state["selected_prompt"],
        placeholder=(
            "Schlage mir einen spielstarken Innenverteidiger für Schalke 04 "
            "mit gutem Spielaufbau vor"
        ),
        label_visibility="collapsed",
    )
    submit_button = st.form_submit_button("🔍 Scouting-Analyse starten")

st.markdown("⚡ **Schnellstart-Beispiele:**")
qc1, qc2, qc3 = st.columns(3)
if qc1.button("🔵 Schalke: IV gesucht"):
    st.session_state["selected_prompt"] = (
        "Schlage mir einen spielstarken Innenverteidiger für Schalke 04 vor"
    )
    st.rerun()
if qc2.button("🟡 BVB: Offensives Talent"):
    st.session_state["selected_prompt"] = (
        "Finde ein junges Offensivtalent mit hohem ROI für Borussia Dortmund"
    )
    st.rerun()
if qc3.button("⚒️ Bochum: Malocher im Mittelfeld"):
    st.session_state["selected_prompt"] = (
        "Ich suche einen zweikampfstarken Sechser mit hohem Malocher-Index für den VfL Bochum"
    )
    st.rerun()

run_analysis = bool(
    submit_button
    or (
        user_prompt
        and user_prompt != st.session_state.get("last_processed", "")
    )
)

# ---------------------------------------------------------
# Parameter Extraction
# ---------------------------------------------------------
if run_analysis:
    if len(user_prompt.strip().split()) < 3:
        st.warning(
            "Deine Anfrage ist sehr kurz. Bitte gib Position, Anforderung "
            "oder Wunschverein an."
        )
    else:
        st.session_state["last_processed"] = user_prompt

        with st.spinner(
            "Malocher Scouting analysiert Vereinsprofil und Datenbank..."
        ):
            extraction_prompt = f"""
Du bist Chef-Scout im Profifußball. Extrahiere die Parameter aus der Anfrage als JSON.

Vereinsprofile, falls keine konkreten Zahlen genannt werden:
- Spitzenvereine (Bayern, BVB, Leipzig, Leverkusen): max_value_eur 50000000, min_overall 78, min_potential 82.
- Ambitionierte Bundesligisten (Frankfurt, Stuttgart, Wolfsburg, Gladbach, Freiburg, Augsburg): max_value_eur 12000000, min_potential 75.
- Mittelfeld / 2. Liga / Traditionsvereine (Schalke, Köln, HSV, Hertha, Bochum, Mainz, St. Pauli): max_value_eur 4000000, max_age 24, min_potential 75.

POSITIONSMAPPING:
- Verteidiger / IV -> CB
- Malocher / Abräumer / Sechser / ZDM -> CDM
- Achter / ZM -> CM
- Zehner / Spielmacher / ZOM -> CAM
- Außenverteidiger / LV / RV -> LB bzw. RB
- Stürmer / MS / Knipser -> ST
- Flügelspieler / LA / RA -> LW bzw. RW

Erlaubte Schlüssel:
club_name, position, max_age, max_value_eur, min_overall,
min_potential, preferred_foot, similar_to_player.

Antworte ausschließlich mit einem validen JSON-Objekt ohne Markdown.
Anfrage: "{user_prompt}"
"""

            raw_text, extraction_error = gemini_generate(extraction_prompt)

            if raw_text:
                try:
                    clean_json = (
                        raw_text.strip()
                        .replace("```json", "")
                        .replace("```", "")
                        .strip()
                    )
                    params = json.loads(clean_json)
                    st.session_state["params"] = params
                    st.session_state["user_prompt"] = user_prompt
                except Exception as exc:
                    st.error(f"Fehler beim Parsen der KI-Kriterien: {exc}")
            else:
                st.error(
                    "Verbindung zum Gemini-Modell fehlgeschlagen. "
                    f"Technische Details: {extraction_error}"
                )

# ---------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------
if "params" in st.session_state:
    params = st.session_state["params"]
    user_prompt = st.session_state["user_prompt"]
    detected_club = params.get("club_name") or "Verein"

    tactical_weights, tactical_labels = get_tactical_weights(user_prompt)

    st.write("**Extrahierte Suchkriterien:**")
    st.json(params)

    st.info(
        "🎯 Taktischer Fokus: "
        + ", ".join(tactical_labels)
        + ". Die Gewichtung basiert auf erkannten Begriffen in der Suchanfrage."
    )

    filtered_df = df.copy()

    max_val = min(
        params.get("max_value_eur") or 999_999_999,
        override_max_value,
    )
    max_a = min(params.get("max_age") or 99, override_max_age)
    min_pot = max(params.get("min_potential") or 0, 0)
    min_pot = min(min_pot, override_min_potential)

    filtered_df = filtered_df[
        (filtered_df["value_eur"] <= max_val)
        & (filtered_df["age"] <= max_a)
        & (filtered_df["potential"] >= min_pot)
    ].copy()

    if params.get("min_overall"):
        filtered_df = filtered_df[
            filtered_df["overall"] >= params["min_overall"]
        ]

    if params.get("preferred_foot") and "preferred_foot" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["preferred_foot"].astype(str).str.contains(
                params["preferred_foot"], na=False, case=False
            )
        ]

    if params.get("position"):
        filtered_df = filtered_df[
            filtered_df["player_positions"].astype(str).str.contains(
                params["position"], na=False, case=False
            )
        ]

    if malocher_mode:
        filtered_df = filtered_df[filtered_df["malocher_index"] >= 70]

    if schnaeppchen_mode:
        filtered_df = filtered_df[filtered_df["roi_score"] >= 1.5]

    if filtered_df.empty:
        st.warning(
            "Keine Spieler gefunden, die alle Kriterien erfüllen. "
            "Lockere die Regler oder formuliere das Profil etwas breiter."
        )
    else:
        filtered_df["tactical_fit"] = calculate_tactical_fit(
            filtered_df, tactical_weights
        )

        # Ähnlichkeit zu Referenzspieler
        similar_to = params.get("similar_to_player")
        target_name = None

        if similar_to:
            match = df[
                df["long_name"].astype(str).str.contains(
                    re.escape(similar_to), case=False, na=False
                )
            ]
            if not match.empty:
                target_player = match.sort_values(
                    by="overall", ascending=False
                ).iloc[[0]]
                target_name = target_player.iloc[0]["long_name"]

                scaler = MinMaxScaler()
                scaled_skills = scaler.fit_transform(
                    filtered_df[skill_columns]
                )
                scaled_target = scaler.transform(
                    target_player[skill_columns]
                )
                sims = cosine_similarity(
                    scaled_skills, scaled_target
                ).flatten()
                filtered_df["match_score_%"] = np.round(sims * 100, 1)
            else:
                filtered_df["match_score_%"] = np.nan
        else:
            filtered_df["match_score_%"] = np.nan

        # Verwendete Gesamtreihung: Tactical Fit + Potenzial + Value
        max_roi = max(float(filtered_df["roi_score"].max()), 0.1)
        filtered_df["roi_norm"] = np.clip(
            filtered_df["roi_score"] / max_roi * 100, 0, 100
        )
        filtered_df["overall_norm"] = np.clip(filtered_df["overall"], 0, 100)
        filtered_df["potential_norm"] = np.clip(filtered_df["potential"], 0, 100)

        if target_name:
            filtered_df["scouting_score"] = np.round(
                filtered_df["tactical_fit"] * 0.40
                + filtered_df["match_score_%"].fillna(0) * 0.25
                + filtered_df["potential_norm"] * 0.20
                + filtered_df["roi_norm"] * 0.10
                + filtered_df["overall_norm"] * 0.05,
                1,
            )
            filtered_df.loc[
                filtered_df["long_name"] == target_name,
                "scouting_score",
            ] = -1
        else:
            filtered_df["scouting_score"] = np.round(
                filtered_df["tactical_fit"] * 0.45
                + filtered_df["potential_norm"] * 0.25
                + filtered_df["roi_norm"] * 0.20
                + filtered_df["overall_norm"] * 0.10,
                1,
            )

        results = filtered_df.sort_values(
            by=["scouting_score", "tactical_fit", "potential"],
            ascending=False,
        ).copy()

        top_matches = results.head(5).copy()
        
        # In Session State speichern für den Chat
        st.session_state["top_matches"] = top_matches
        st.session_state["tactical_labels"] = tactical_labels

        # -----------------------------------------------------
        # Top Recommendation
        # -----------------------------------------------------
        top_player = top_matches.iloc[0]
        st.markdown("### 🏆 Top-Empfehlung")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Spieler", top_player["long_name"], help="Name des empfohlenen Top-Kandidaten")
        m2.metric(
            "OVR / POT",
            f"{int(top_player['overall'])} / {int(top_player['potential'])}",
            help="OVR = Overall Rating (Aktuelle Gesamtstärke) | POT = Potential (Erwartete maximale Gesamtstärke)"
        )
        m3.metric(
            "🎯 Tactical Fit", 
            f"{top_player['tactical_fit']:.1f}", 
            help="Taktische Passung (0-100) der Spieler-Skills zum erkannten Anforderungsprofil."
        )
        m4.metric(
            "⚒️ Malocher-Index", 
            f"{top_player['malocher_index']:.1f}", 
            help="Arbeits- und Einsatzindex, berechnet aus: Physis (45%) + Defensive (35%) + Tempo (20%)."
        )
        m5.metric(
            "💎 ROI-Faktor", 
            f"{top_player['roi_score']:.2f}", 
            help="Return on Investment: Verhältnis aus Entwicklungspotenzial (POT - OVR) zum aktuellen Marktwert in Mio. €."
        )

        # -----------------------------------------------------
        # Explainability
        # -----------------------------------------------------
        st.markdown("### 🧠 Warum dieser Spieler?")
        explanation = []
        if top_player["tactical_fit"] >= 80:
            explanation.append(
                f"Hohe taktische Passung ({top_player['tactical_fit']:.1f}/100) zum erkannten Profil."
            )
        elif top_player["tactical_fit"] >= 70:
            explanation.append(
                f"Solide taktische Passung ({top_player['tactical_fit']:.1f}/100) zum erkannten Profil."
            )
        if top_player["potential_growth"] >= 8:
            explanation.append(
                f"Großes Entwicklungspotenzial von +{int(top_player['potential_growth'])} OVR-Punkten."
            )
        elif top_player["potential_growth"] >= 4:
            explanation.append(
                f"Ordentliches Entwicklungspotenzial von +{int(top_player['potential_growth'])} OVR-Punkten."
            )
        if top_player["malocher_index"] >= 80:
            explanation.append(
                f"Sehr ausgeprägtes Malocher-Profil ({top_player['malocher_index']:.1f})."
            )
        if top_player["roi_score"] >= 3:
            explanation.append(
                f"Sehr attraktiver ROI-Faktor ({top_player['roi_score']:.2f}) im Verhältnis von Entwicklungspotenzial zu Marktwert."
            )
        if top_player["value_eur"] <= 5_000_000:
            explanation.append(
                f"Marktwert liegt bei rund {top_player['value_eur'] / 1_000_000:.1f} Mio. €."
            )

        if explanation:
            for reason in explanation[:5]:
                st.write(f"• {reason}")
        else:
            st.write("Die Empfehlung basiert primär auf Tactical Fit, Potenzial, ROI und OVR.")

        # -----------------------------------------------------
        # Top 5 Table + Shortlist
        # -----------------------------------------------------
        st.markdown("### 📋 Top 5 Kandidaten")
        table_columns = [
            "long_name",
            "age",
            "player_positions",
            "overall",
            "potential",
            "value_eur",
            "tactical_fit",
            "malocher_index",
            "roi_score",
            "match_score_%",
            "scouting_score",
        ]
        table_df = top_matches[table_columns].copy()
        table_df["value_eur"] = table_df["value_eur"].apply(
            lambda x: f"{x / 1_000_000:.2f} Mio. €"
        )
        table_df = table_df.rename(
            columns={
                "long_name": "Spieler",
                "age": "Alter",
                "player_positions": "Position",
                "overall": "OVR",
                "potential": "POT",
                "value_eur": "Marktwert",
                "tactical_fit": "🎯 Tactical Fit",
                "malocher_index": "⚒️ Malocher",
                "roi_score": "💎 ROI",
                "match_score_%": "Match-Score",
                "scouting_score": "Scouting Score",
            }
        )
        
        st.dataframe(
            table_df, 
            column_config={
                "Spieler": st.column_config.TextColumn("Spieler", help="Vollständiger Name des Spielers"),
                "Alter": st.column_config.NumberColumn("Alter", help="Alter in Jahren"),
                "Position": st.column_config.TextColumn("Position", help="Spielpositionen (z.B. CB, CDM, ST)"),
                "OVR": st.column_config.NumberColumn("OVR", help="Overall Rating: Aktuelle Gesamtstärke (0-99)"),
                "POT": st.column_config.NumberColumn("POT", help="Potential: Erwartete maximale Gesamtstärke (0-99)"),
                "Marktwert": st.column_config.TextColumn("Marktwert", help="Aktueller Marktwert in Millionen Euro"),
                "🎯 Tactical Fit": st.column_config.NumberColumn("🎯 Tactical Fit", help="Taktische Passung (0-100) zum Anforderungsprofil"),
                "⚒️ Malocher": st.column_config.NumberColumn("⚒ Malocher", help="Malocher-Index: Physis (45%) + Defensive (35%) + Tempo (20%)"),
                "💎 ROI": st.column_config.NumberColumn("💎 ROI", help="Return on Investment: Potenzial-Wachstum geteilt durch Marktwert in Mio. €"),
                "Match-Score": st.column_config.NumberColumn("Match-Score", help="Ähnlichkeit in % zu einem gesuchten Referenzspieler (Cosine Similarity)"),
                "Scouting Score": st.column_config.NumberColumn("Scouting Score", help="Gesamt-Priorisierungsscore der KI aus Tactical Fit, Potenzial, ROI und OVR"),
            },
            use_container_width=True, 
            hide_index=True
        )

        st.caption(
            "Scouting Score kombiniert Tactical Fit, Ähnlichkeit (falls vorhanden), "
            "Potenzial, ROI und OVR. Er dient als technische Priorisierung und ersetzt keine menschliche Bewertung."
        )

        save_cols = st.columns(len(top_matches))
        for idx, (_, row) in enumerate(top_matches.iterrows()):
            player_name = row["long_name"]
            is_saved = player_name in st.session_state["shortlist"]
            label = "✓ Gespeichert" if is_saved else "⭐ Auf Shortlist"
            if save_cols[idx].button(label, key=f"save_{player_name}_{idx}"):
                if is_saved:
                    st.session_state["shortlist"].remove(player_name)
                else:
                    st.session_state["shortlist"].append(player_name)
                st.rerun()

        # -----------------------------------------------------
        # Hidden Gems
        # -----------------------------------------------------
        st.markdown("### 💎 Hidden Gems")
        hidden_pool = filtered_df.copy()
        hidden_pool = hidden_pool.sort_values(
            by=["roi_score", "potential_growth", "tactical_fit"],
            ascending=False,
        )
        hidden_pool = hidden_pool[
            (hidden_pool["potential_growth"] >= 5)
            & (hidden_pool["value_eur"] <= override_max_value)
        ].head(5)

        if hidden_pool.empty:
            st.caption("Unter den aktuellen Filtern wurden keine Hidden Gems gefunden.")
        else:
            hg_display = hidden_pool[
                [
                    "long_name",
                    "age",
                    "overall",
                    "potential",
                    "value_eur",
                    "tactical_fit",
                    "roi_score",
                ]
            ].copy()
            hg_display["value_eur"] = hg_display["value_eur"].apply(
                lambda x: f"{x / 1_000_000:.2f} Mio. €"
            )
            hg_display = hg_display.rename(
                columns={
                    "long_name": "Spieler",
                    "age": "Alter",
                    "overall": "OVR",
                    "potential": "POT",
                    "value_eur": "Marktwert",
                    "tactical_fit": "Tactical Fit",
                    "roi_score": "ROI",
                }
            )
            st.dataframe(
                hg_display, 
                column_config={
                    "Spieler": st.column_config.TextColumn("Spieler", help="Name des Talents"),
                    "Alter": st.column_config.NumberColumn("Alter", help="Alter in Jahren"),
                    "OVR": st.column_config.NumberColumn("OVR", help="Aktuelle Gesamtstärke (Overall)"),
                    "POT": st.column_config.NumberColumn("POT", help="Erwartetes Maximalpotenzial"),
                    "Marktwert": st.column_config.TextColumn("Marktwert", help="Aktueller Marktwert in Mio. €"),
                    "Tactical Fit": st.column_config.NumberColumn("Tactical Fit", help="Taktische Passung (0-100)"),
                    "ROI": st.column_config.NumberColumn("ROI", help="Verhältnis von Potenzialwachstum zu Marktwert"),
                },
                use_container_width=True, 
                hide_index=True
            )

        # -----------------------------------------------------
        # Multi-player Scouting Battle
        # -----------------------------------------------------
        st.markdown("### 🥊 Scouting Battle – 3 bis 5 Spieler")
        comparison_list = top_matches["long_name"].tolist()
        selected_players = st.multiselect(
            "Spieler für den direkten Vergleich auswählen",
            options=comparison_list,
            default=comparison_list[: min(3, len(comparison_list))],
            max_selections=5,
        )

        if len(selected_players) >= 2:
            compare_df = top_matches[
                top_matches["long_name"].isin(selected_players)
            ].copy()
            compare_df = compare_df[
                [
                    "long_name",
                    "overall",
                    "potential",
                    "age",
                    "tactical_fit",
                    "malocher_index",
                    "roi_score",
                    "value_eur",
                ]
            ].set_index("long_name").T
            compare_df.index = [
                "OVR",
                "POT",
                "Alter",
                "Tactical Fit",
                "Malocher-Index",
                "ROI-Faktor",
                "Marktwert (€)",
            ]
            st.dataframe(compare_df, use_container_width=True)
            st.caption("ℹ️ **Kennzahlen-Erklärung:** **OVR** = Aktuelle Stärke | **POT** = Potenzial | **Tactical Fit** = Taktische Passung | **Malocher-Index** = Physis/Defensive/Tempo-Wert | **ROI-Faktor** = Potenzialwachstum pro Mio. € Marktwert.")

            # Ein-/ausblendbares Säulendiagramm (Gruppiertes Bar Chart)
            with st.expander("📊 Skill-Vergleich (Säulendiagramm) anzeigen", expanded=False):
                radar_names = selected_players[:5]
                categories = skill_columns
                labels = [SKILL_MAP[c].upper() for c in categories]

                x = np.arange(len(categories))
                num_players = len(radar_names)
                width = 0.8 / num_players

                fig, ax = plt.subplots(figsize=(9, 5))
                fig.patch.set_facecolor("#0e1117")
                ax.set_facecolor("#161b26")
                ax.tick_params(colors="white", labelsize=9)
                ax.spines["bottom"].set_color("white")
                ax.spines["left"].set_color("white")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="white")
                ax.set_axisbelow(True)

                colors = ["#004D98", "#E30613", "#28a745", "#ffc107", "#17a2b8"]

                for i, player_name in enumerate(radar_names):
                    player_row = df[df["long_name"] == player_name].iloc[0]
                    values = [float(player_row[c]) for c in categories]
                    offset = (i - num_players / 2 + 0.5) * width
                    ax.bar(
                        x + offset,
                        values,
                        width,
                        label=player_name,
                        color=colors[i % len(colors)],
                        alpha=0.9,
                    )

                ax.set_xticks(x)
                ax.set_xticklabels(labels, color="white", fontweight="bold")
                ax.set_ylabel("Skill-Wert (0-100)", color="white", fontsize=10)
                ax.set_ylim(0, 105)
                ax.tick_params(axis="y", colors="white")

                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.18),
                    ncol=min(num_players, 3),
                    facecolor="#161b26",
                    edgecolor="none",
                    labelcolor="white",
                    fontsize=9,
                )
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.caption("Bitte mindestens zwei Spieler auswählen.")

        # -----------------------------------------------------
        # AI Scouting Report
        # -----------------------------------------------------
        st.markdown(f"### 📝 Scouting-Bericht für {detected_club}")
        report_prompt = f"""
Du bist Chef-Scout bei {detected_club}.
Anfrage des Managements: "{user_prompt}"

Erkannter taktischer Fokus: {', '.join(tactical_labels)}

Top-Kandidaten mit berechneten Kennzahlen:
{top_matches.to_string(index=False)}

Schreibe einen professionellen Scouting-Bericht.
Struktur:
1. Kurzfazit
2. Warum der Top-Kandidat zum Profil passt
3. Vergleich mit den weiteren Kandidaten
4. Entwicklungspotenzial und Value for Money
5. Risiken / offene Punkte
6. Sinnvolle nächste Schritte für das Scouting

Verwende nur die gelieferten Daten. Erfinde keine Spielpraxis, Verletzungen,
Gehaltsdaten, Ablösen oder sonstige Informationen, die nicht in den Daten stehen.
"""
        rep_text, report_error = gemini_generate(report_prompt)

        if rep_text:
            st.markdown(rep_text)
            try:
                pdf_bytes = create_pdf_report(
                    detected_club,
                    user_prompt,
                    rep_text,
                    top_matches,
                )
                st.download_button(
                    label="📄 Scouting-Bericht als PDF herunterladen",
                    data=pdf_bytes,
                    file_name=f"Scouting_Bericht_{detected_club}.pdf",
                    mime="application/pdf",
                )
            except Exception as pdf_error:
                st.warning(f"PDF-Erstellung fehlgeschlagen: {pdf_error}")
        else:
            st.warning(
                "Der KI-Bericht konnte nicht erzeugt werden. "
                f"Details: {report_error}"
            )
else:
    st.info(
        "Starte eine Scouting-Analyse oder nutze einen der Schnellstart-Buttons."
    )

# ---------------------------------------------------------
# Malocher Scout Chat (IMMER VERFÜGBAR AUCH AUF STARTSEITE)
# ---------------------------------------------------------
st.divider()
st.subheader("💬 Frage den Malocher Scout")

for q, a in st.session_state["chat_history"]:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)

user_question = st.chat_input(
    "Frage den Malocher Scout... (z. B. Was weißt du über Spieler X? Oder: Passt Spieler X zu Schalke?)"
)

if user_question:
    st.chat_message("user").write(user_question)

    # 1. Automatische Erkennung & Suche nach genannten Spielern im Gesamtdatensatz (df)
    mentioned_players_data = ""
    for name in df["long_name"].dropna().unique():
        # Suche nach signifikanten Namensteilen (> 3 Zeichen)
        name_parts = [p for p in str(name).split() if len(p) > 3]
        if any(part.lower() in user_question.lower() for part in name_parts):
            player_row = df[df["long_name"] == name].iloc[0]
            mentioned_players_data += (
                f"\n- Gefundener Spieler in Datenbank: {player_row['long_name']} | "
                f"Alter: {player_row['age']} | OVR: {player_row['overall']} | "
                f"POT: {player_row['potential']} | Marktwert: {player_row['value_eur']/1e6:.2f} Mio. € | "
                f"Position: {player_row['player_positions']} | Malocher-Index: {player_row['malocher_index']} | "
                f"ROI: {player_row['roi_score']} | Skills (Tempo:{player_row['pace']}, Schuss:{player_row['shooting']}, Passen:{player_row['passing']}, Dribbling:{player_row['dribbling']}, Def:{player_row['defending']}, Physis:{player_row['physic']})\n"
            )
            break

    # 2. Kontext für Top-Kandidaten aus der Session holen (falls vorhanden)
    top_matches_session = st.session_state.get("top_matches")
    if top_matches_session is not None and not top_matches_session.empty:
        top_context = top_matches_session[['long_name', 'age', 'overall', 'potential', 'value_eur', 'player_positions', 'malocher_index', 'roi_score']].to_string(index=False)
    else:
        top_context = "Aktuell wurde noch keine spezifische Suchanalyse gestartet."

    detected_club = st.session_state.get("params", {}).get("club_name") or "einem Verein"
    tactical_labels_session = st.session_state.get("tactical_labels", ["Ausgewogen"])

    chat_prompt = f"""
Du bist der "Malocher Scout", ein erfahrener, datengestützter Profi-Scout im Fußball.
Der Manager fragt dich Folgendes: "{user_question}"

Kontext aus der aktuellen Suchanalyse (falls bereits gelaufen):
{top_context}

Zusätzlich aus der Gesamtdatenbank erkannter Spieler (falls zutreffend):
{mentioned_players_data if mentioned_players_data else "Kein spezifischer Spieler in der Frage direkt erkannt."}

Taktischer Fokus der aktuellen Analyse (falls vorhanden): {', '.join(tactical_labels_session)}

Anweisungen:
- Du bist der "Malocher Scout". Antworte kompetent, direkt, praxisnah und auf Deutsch.
- Wenn nach einem speziellen Spieler gefragt wird ("Was weißt du über X?" / "Passt X zu Y?"), nutze die Datenbank-Informationen und beurteile Marktwert, Potenzial, Alter, Skills und Position im Kontext des Zielvereins.
- Wenn noch keine Suche gelaufen ist, beantworte allgemeine Fragen oder Spieleranfragen direkt auf Basis der gelieferten Daten und deines Fachwissens.
- Antworte professionell, präzise und ausschließlich auf Basis der vorliegenden Daten.
"""
    chat_reply, chat_error = gemini_generate(chat_prompt)

    if chat_reply:
        st.chat_message("assistant").write(chat_reply)
        st.session_state["chat_history"].append(
            (user_question, chat_reply)
        )
        st.rerun()
    else:
        st.error(f"Malocher Scout konnte nicht antworten: {chat_error}")

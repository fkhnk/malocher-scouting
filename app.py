import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

st.set_page_config(page_title="Glückauf Scouting 04", layout="wide")

st.title("Malocher Scouting")
st.markdown("Universelle, datengestützte Spielersuche für Profivereine powered by **Gemini & Cosine Similarity**")

# API Key sichern (aus Secrets oder direkt)
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    GEMINI_API_KEY = "AQ.Ab8RN6Iy8TKRCVtE_BUHCalfZZwiqynPV4_TFGo536gX6SW48A"

client = genai.Client(api_key=GEMINI_API_KEY)
FALLBACK_MODELS = ['gemini-3.8-flash', 'gemini-3.5-flash', 'gemini-flash-latest']

# Übersetzung der Skill-Spalten für das Spinnendiagramm
SKILL_MAP = {
    'pace': 'Tempo',
    'shooting': 'Schuss',
    'passing': 'Passen',
    'dribbling': 'Dribbling',
    'defending': 'Defensive',
    'physic': 'Physis'
}

@st.cache_data
def load_data():
    df = pd.read_csv('FC26_20250921.csv', low_memory=False)
    skill_columns = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
    df[skill_columns] = df[skill_columns].fillna(df[skill_columns].mean())
    return df, skill_columns

df, skill_columns = load_data()

user_prompt = st.text_input(
    "Welches Profil suchst du und für welchen Verein?", 
    "Schlage mir einen spielstarken Innenverteidiger für Schalke 04 vor"
)

if st.button("🔍 Scouting-Analyse starten"):
    with st.spinner("Glückauf Scouting analysiert das Vereinsprofil und durchsucht die Datenbank..."):
        
        extraction_prompt = f"""
        Du bist ein weltklasse Chef-Scout im Profifußball. Extrahiere die Parameter als JSON aus der Anfrage.
        
        WICHTIG - DYNAMISCHE VEREINSERKENNUNG & IMPLIZITE CLUB-PROFILE:
        Analysiere, welcher Verein in der Anfrage genannt wird (z. B. FC Bayern, BVB, Eintracht Frankfurt, Schalke 04, Werder Bremen, HSV, Köln, Stuttgart, etc.).
        Falls der User keine konkreten Zahlen (Budget/Alter) nennt, wende automatisch ein passendes reales Vereinsprofil an:
        - Spitzenvereine (z. B. Bayern, BVB, Leipzig, Leverkusen): max_value_eur: 50000000, min_overall: 78, min_potential: 82.
        - Ambitionierte Bundesligisten (z. B. Frankfurt, Stuttgart, Wolfsburg, Gladbach, Freiburg): max_value_eur: 15000000, min_potential: 78.
        - Mittelfeld / 2. Liga / Traditionsvereine (z. B. Schalke, Köln, HSV, Hertha, Bochum, Mainz, St. Pauli): max_value_eur: 4000000, max_age: 24, min_potential: 75.

        POSITIONSMAPPING (Deutsch/Jargon -> EA FC Englisch):
        - Malocher / Abräumer / Sechser / ZDM / DM -> 'CDM'
        - Achter / ZM -> 'CM'
        - Zehner / Spielmacher / ZOM -> 'CAM'
        - Innenverteidiger / IV -> 'CB'
        - Außenverteidiger / LV / RV -> 'LB' bzw. 'RB'
        - Stürmer / MS / Knipser -> 'ST'
        - Flügelspieler / LA / RA -> 'LW' bzw. 'RW'

        Erlaubte JSON-Schlüssel:
        - club_name (String, z. B. 'Eintracht Frankfurt', 'FC Bayern München', 'FC Schalke 04' oder 'Allgemein')
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
                res = client.models.generate_content(model=m, contents=extraction_prompt)
                raw_text = res.text
                break
            except Exception:
                continue
                
        if raw_text:
            clean_json = raw_text.strip().replace('```json', '').replace('```', '')
            params = json.loads(clean_json)
            
            st.write("**Extrahierte Kriterien (Inkl. Vereinsprofil):**")
            st.json(params)
            
            filtered_df = df.copy()
            if params.get('max_age'):
                filtered_df = filtered_df[filtered_df['age'] <= params['max_age']]
            if params.get('max_value_eur'):
                filtered_df = filtered_df[filtered_df['value_eur'] <= params['max_value_eur']]
            if params.get('min_overall'):
                filtered_df = filtered_df[filtered_df['overall'] >= params['min_overall']]
            if params.get('min_potential'):
                filtered_df = filtered_df[filtered_df['potential'] >= params['min_potential']]
            if params.get('preferred_foot'):
                filtered_df = filtered_df[filtered_df['preferred_foot'].str.contains(params['preferred_foot'], na=False, case=False)]
            if params.get('position'):
                filtered_df = filtered_df[filtered_df['player_positions'].str.contains(params['position'], na=False, case=False)]
            
            if filtered_df.empty:
                st.warning("Keine Spieler in der Datenbank gefunden, die alle Kriterien erfüllen! Versuche die Filter etwas aufzuweichen.")
            else:
                similar_to = params.get('similar_to_player')
                target_name = None
                
                if similar_to:
                    match = df[df['short_name'].str.contains(similar_to, case=False, na=False)]
                    if not match.empty:
                        target_player = match.sort_values(by='overall', ascending=False).iloc[[0]]
                        target_name = target_player.iloc[0]['short_name']
                        
                        scaler = MinMaxScaler()
                        scaled_skills = scaler.fit_transform(filtered_df[skill_columns])
                        scaled_target = scaler.transform(target_player[skill_columns])
                        
                        sims = cosine_similarity(scaled_skills, scaled_target).flatten()
                        filtered_df['match_score_%'] = np.round(sims * 100, 1)
                        results = filtered_df[filtered_df['short_name'] != target_name].sort_values(by='match_score_%', ascending=False)
                    else:
                        filtered_df['match_score_%'] = "-"
                        results = filtered_df.sort_values(by=['potential', 'overall'], ascending=False)
                else:
                    filtered_df['match_score_%'] = "-"
                    results = filtered_df.sort_values(by=['potential', 'overall'], ascending=False)
                    
                top_matches = results[['short_name', 'age', 'player_positions', 'overall', 'potential', 'value_eur', 'match_score_%']].head(5)
                
                display_matches = top_matches.rename(columns={
                    'short_name': 'Name',
                    'age': 'Alter',
                    'player_positions': 'Positionen',
                    'overall': 'Gesamtstärke (OVR)',
                    'potential': 'Potenzial (POT)',
                    'value_eur': 'Marktwert (€)',
                    'match_score_%': 'Match-Score (%)'
                })
                
                detected_club = params.get('club_name', 'Verein')
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader(f"📋 Top 5 Treffer für {detected_club}")
                    st.dataframe(display_matches, use_container_width=True)
                    
                with col2:
                    german_skill_labels = [SKILL_MAP[col].upper() for col in skill_columns]
                    
                    if target_name and not top_matches.empty:
                        st.subheader("📊 Radar-Vergleich")
                        p1 = df[df['short_name'] == target_name].iloc[0]
                        p2 = df[df['short_name'] == top_matches.iloc[0]['short_name']].iloc[0]
                        
                        categories = skill_columns
                        v1 = p1[categories].values.tolist() + [p1[categories].values[0]]
                        v2 = p2[categories].values.tolist() + [p2[categories].values[0]]
                        angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))] + [0]
                        
                        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
                        plt.xticks(angles[:-1], german_skill_labels)
                        ax.plot(angles, v1, label=p1['short_name'], color='#004D98')
                        ax.plot(angles, v2, label=p2['short_name'], color='#E30613')
                        ax.fill(angles, v1, alpha=0.1, color='#004D98')
                        ax.fill(angles, v2, alpha=0.1, color='#E30613')
                        plt.legend(loc='lower right')
                        st.pyplot(fig)
                    elif not top_matches.empty:
                        st.subheader("📊 Fähigkeitsprofil des Top-Treffers")
                        top_p = df[df['short_name'] == top_matches.iloc[0]['short_name']].iloc[0]
                        categories = skill_columns
                        v_top = top_p[categories].values.tolist() + [top_p[categories].values[0]]
                        angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))] + [0]
                        
                        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
                        plt.xticks(angles[:-1], german_skill_labels)
                        ax.plot(angles, v_top, label=top_p['short_name'], color='#004D98')
                        ax.fill(angles, v_top, alpha=0.1, color='#004D98')
                        plt.legend(loc='lower right')
                        st.pyplot(fig)
                
                st.subheader(f"📝 Scouting-Bericht für {detected_club}")
                report_prompt = f"""
                Du bist Chef-Scout bei {detected_club}. 
                Der Trainer/Manager hat angefragt: '{user_prompt}'. 
                
                Hier sind die datenbasierten Top-Kandidaten aus der Datenbank:
                {top_matches.to_string()}
                
                Schreibe einen professionellen Scouting-Bericht direkt an die Vereinsführung von {detected_club}.
                Begründe, warum diese Spieler perfekt zur sportlichen und finanziellen Philosophie von {detected_club} passen.
                """
                rep_text = None
                for m in FALLBACK_MODELS:
                    try:
                        rep_text = client.models.generate_content(model=m, contents=report_prompt).text
                        break
                    except Exception:
                        continue
                if rep_text:
                    st.markdown(rep_text)

import io
import re
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import streamlit as st
from streamlit_sortables import sort_items  # Neues Drag-and-Drop-Modul

# Erweiterte Pflichtspalten inkl. Alter und Disziplin
REQUIRED_COLUMNS = [
    "Name",
    "Verein",
    "Geschlecht",
    "Alter",
    "Gewicht",
    "Disziplin",
    "Kämpfe",
    "Siege",
    "Niederlagen",
]

COLUMN_ALIASES = {
    "name": "Name", "kaempfer": "Name", "kämpfer": "Name", "fighter": "Name", "vorname nachname": "Name",
    "verein": "Verein", "club": "Verein", "gym": "Verein", "team": "Verein",
    "geschlecht": "Geschlecht", "gender": "Geschlecht", "sex": "Geschlecht",
    "alter": "Alter", "age": "Alter",
    "gewicht": "Gewicht", "gewicht kg": "Gewicht", "gewicht (kg)": "Gewicht", "kg": "Gewicht",
    "disziplin": "Disziplin", "discipline": "Disziplin", "regelwerk": "Disziplin", "art": "Disziplin",
    "kaempfe": "Kämpfe", "kämpfe": "Kämpfe", "anzahl kaempfe": "Kämpfe", "anzahl kämpfe": "Kämpfe",
    "anzahl an kämpfen": "Kämpfe", "fights": "Kämpfe", "total fights": "Kämpfe",
    "siege": "Siege", "gewonnen": "Siege", "gewonnene kaempfe": "Siege", "gewonnene kämpfe": "Siege", "wins": "Siege",
    "niederlagen": "Niederlagen", "verloren": "Niederlagen", "verlorene kaempfe": "Niederlagen", "verlorene kämpfe": "Niederlagen", "losses": "Niederlagen",
}


@dataclass
class Weights:
    weight: float; age: float; fights: float; wins: float; losses: float; winrate: float

@dataclass
class MatchSettings:
    max_weight_diff: float; max_age_diff: int; max_fight_diff: int; max_win_diff: int; max_loss_diff: int
    min_score: float; same_gender_only: bool; strict_class_matching: bool; strict_weight_class: bool
    strict_discipline_matching: bool; weights: Weights


def normalize_header(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[\n\r\t_\-/]+", " ", text)
    return re.sub(r"\s+", " ", text)

def normalize_gender(value: object) -> str:
    text = str(value).strip().lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    if text in {"m", "maennlich", "mann", "male", "herr", "junge"}: return "männlich"
    if text in {"w", "weiblich", "frau", "female", "dame", "maedchen"}: return "weiblich"
    return "divers" if text in {"d", "divers", "nonbinary", "nb"} else str(value).strip()

def normalize_discipline(value: object) -> str:
    text = str(value).strip().lower().replace(" ", "").replace("-", "")
    if "k1" in text or "kick" in text: return "K-1"
    if "muay" in text or "thai" in text or "boxen" in text: return "Muay Thai"
    return str(value).strip()

def normalize_club(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())

def get_experience_class(fights: int) -> str:
    if fights == 0: return "Newcomer (0)"
    elif fights <= 5: return "D-Klasse (1-5)"
    elif fights <= 15: return "C-Klasse (6-15)"
    elif fights <= 25: return "B-Klasse (16-25)"
    return "A-Klasse (>25)"

def get_weight_class(weight: float, weight_classes: List[Tuple[str, float]]) -> str:
    for name, limit in sorted(weight_classes, key=lambda x: x[1]):
        if weight <= limit: return f"{name} (-{limit}kg)"
    return f"Superschwergewicht (> {sorted(weight_classes, key=lambda x: x[1])[-1][1]}kg)" if weight_classes else "Unbekannt"

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    used_targets = set()
    for col in df.columns:
        key = normalize_header(col)
        key_ascii = key.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
        target = COLUMN_ALIASES.get(key) or COLUMN_ALIASES.get(key_ascii)
        if target and target not in used_targets:
            rename_map[col] = target; used_targets.add(target)
    return df.rename(columns=rename_map)

def clean_fighters(raw_df: pd.DataFrame, weight_classes: List[Tuple[str, float]]) -> Tuple[pd.DataFrame, List[str]]:
    df = normalize_columns(raw_df).copy()
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing: raise ValueError("Diese Pflichtspalten fehlen: " + ", ".join(missing))
    df = df[REQUIRED_COLUMNS].copy().dropna(how="all")

    for col in ["Name", "Verein", "Geschlecht", "Disziplin"]: df[col] = df[col].astype(str).str.strip()
    for col in ["Alter", "Gewicht", "Kämpfe", "Siege", "Niederlagen"]: df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Geschlecht_norm"] = df["Geschlecht"].apply(normalize_gender)
    df["Disziplin_norm"] = df["Disziplin"].apply(normalize_discipline)
    df["Verein_norm"] = df["Verein"].apply(normalize_club)
    df["Winrate"] = df.apply(lambda r: float(r["Siege"]) / float(r["Kämpfe"]) if r["Kämpfe"] and r["Kämpfe"] > 0 else 0.0, axis=1)

    errors: List[str] = []
    for idx, row in df.iterrows():
        line = idx + 2
        if not row["Name"] or row["Name"].lower() == "nan": errors.append(f"Zeile {line}: Name fehlt.")
        if not row["Verein"] or row["Verein"].lower() == "nan": errors.append(f"Zeile {line}: Verein fehlt.")
        if not row["Disziplin"] or row["Disziplin"].lower() == "nan": errors.append(f"Zeile {line}: Disziplin fehlt.")
        for col in ["Alter", "Gewicht", "Kämpfe", "Siege", "Niederlagen"]:
            if pd.isna(row[col]): errors.append(f"Zeile {line}: {col} ist keine Zahl.")
    if errors: return df, errors

    df["Alter"] = df["Alter"].round().astype(int)
    df["Kämpfe"] = df["Kämpfe"].round().astype(int)
    df["Siege"] = df["Siege"].round().astype(int)
    df["Niederlagen"] = df["Niederlagen"].round().astype(int)
    df["Gewicht"] = df["Gewicht"].astype(float)
    df["Klasse"] = df["Kämpfe"].apply(get_experience_class)
    df["Gewichtsklasse"] = df["Gewicht"].apply(lambda w: get_weight_class(w, weight_classes))
    return df.reset_index(drop=True), []

def score_pair(a: pd.Series, b: pd.Series, settings: MatchSettings) -> Optional[Dict[str, float]]:
    if a["Verein_norm"] == b["Verein_norm"]: return None
    if settings.same_gender_only and a["Geschlecht_norm"] != b["Geschlecht_norm"]: return None
    if settings.strict_class_matching and a["Klasse"] != b["Klasse"]: return None
    if settings.strict_weight_class and a["Gewichtsklasse"] != b["Gewichtsklasse"]: return None
    if settings.strict_discipline_matching and a["Disziplin_norm"] != b["Disziplin_norm"]: return None

    weight_diff = abs(float(a["Gewicht"]) - float(b["Gewicht"]))
    age_diff = abs(int(a["Alter"]) - int(b["Alter"]))
    fight_diff = abs(int(a["Kämpfe"]) - int(b["Kämpfe"]))
    win_diff = abs(int(a["Siege"]) - int(b["Siege"]))
    loss_diff = abs(int(a["Niederlagen"]) - int(b["Niederlagen"]))
    winrate_diff = abs(float(a["Winrate"]) - float(b["Winrate"]))

    if weight_diff > settings.max_weight_diff or age_diff > settings.max_age_diff or fight_diff > settings.max_fight_diff: return None

    weights_sum = sum([settings.weights.weight, settings.weights.age, settings.weights.fights, settings.weights.wins, settings.weights.losses, settings.weights.winrate])
    weights_sum = weights_sum if weights_sum > 0 else 1.0

    penalty = (settings.weights.weight * min(weight_diff / max(settings.max_weight_diff, 0.1), 1.0) +
               settings.weights.age * min(age_diff / max(settings.max_age_diff, 1), 1.0) +
               settings.weights.fights * min(fight_diff / max(settings.max_fight_diff, 1), 1.0))
    
    score = max(0.0, 100.0 * (1.0 - penalty / weights_sum))
    if a["Disziplin_norm"] != b["Disziplin_norm"]: score = max(0.0, score - 15.0)
    return {"Score": round(score, 1), "Gewichtsdifferenz": round(weight_diff, 2), "Altersdifferenz": age_diff, "Kampfdifferenz": fight_diff} if score >= settings.min_score else None

def build_matches(fighters: pd.DataFrame, settings: MatchSettings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    graph = nx.Graph()
    for idx in fighters.index: graph.add_node(idx)
    for i in fighters.index:
        for j in fighters.index:
            if j <= i: continue
            details = score_pair(fighters.loc[i], fighters.loc[j], settings)
            if details:
                w = int(details["Score"] * 10000) - int(details["Gewichtsdifferenz"] * 10)
                graph.add_edge(i, j, weight=w, details=details)

    matching = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True, weight="weight")
    match_rows = []
    matched_indices = set()
    for idx, (i, j) in enumerate(sorted(matching, key=lambda p: -graph.edges[p]["details"]["Score"]), 1):
        details = graph.edges[i, j]["details"]
        match_rows.append(make_output_row(idx, fighters.loc[i], fighters.loc[j], details))
        matched_indices.update([i, j])

    unmatched_rows = [
        {"ID": idx, "Name": r["Name"], "Verein": r["Verein"], "Geschlecht": r["Geschlecht_norm"], "Alter": r["Alter"], "Gewicht": r["Gewicht"], "Gewichtsklasse": r["Gewichtsklasse"], "Disziplin": r["Disziplin_norm"], "Kämpfe": r["Kämpfe"], "Klasse": r["Klasse"]}
        for idx, r in fighters.iterrows() if idx not in matched_indices
    ]
    return pd.DataFrame(match_rows), pd.DataFrame(unmatched_rows)

def make_output_row(nr: int, a: pd.Series, b: pd.Series, details: Dict[str, float]) -> Dict[str, object]:
    disc = a["Disziplin_norm"] if a["Disziplin_norm"] == b["Disziplin_norm"] else f"{a['Disziplin_norm']} vs {b['Disziplin_norm']}"
    # Generiere eine eindeutige ID für das Drag-and-Drop-System
    unique_id = f"KAMPF: {a['Name']} ({a['Verein']}) vs {b['Name']} ({b['Verein']}) [{disc}]"
    return {
        "id": unique_id, "Kampf Nr.": nr, "Kampf-Disziplin": disc,
        "Kämpfer A": a["Name"], "Verein A": a["Verein"], "Gewicht A": a["Gewicht"], "Kämpfe A": a["Kämpfe"],
        "Kämpfer B": b["Name"], "Verein B": b["Verein"], "Gewicht B": b["Gewicht"], "Kämpfe B": b["Kämpfe"],
        **details
    }

def export_to_excel(matches: pd.DataFrame, unmatched: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        matches.to_excel(writer, sheet_name="Matches", index=False)
        unmatched.to_excel(writer, sheet_name="Nicht gematcht", index=False)
    return output.getvalue()

def template_excel() -> bytes:
    sample = pd.DataFrame([
        ["Kämpfer 1", "Gym Alpha", "männlich", 25, 70.0, "Muay Thai", 3, 2, 1],
        ["Kämpfer 2", "Gym Beta", "männlich", 27, 71.2, "Muay Thai", 4, 2, 2],
        ["Kämpfer 3", "Gym Gamma", "weiblich", 19, 58.5, "K-1", 1, 1, 0],
        ["Kämpfer 4", "Gym Delta", "weiblich", 21, 59.0, "K-1", 2, 1, 1]
    ], columns=REQUIRED_COLUMNS)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer: sample.to_excel(writer, sheet_name="Kämpfer", index=False)
    return output.getvalue()

# Streamlit Setup
st.set_page_config(page_title="Muay Thai & K1 Matchmaker", page_icon="🥊", layout="wide")
st.title("🥊 Muay Thai & K1 Matchmaker Pro")

if "weight_classes" not in st.session_state:
    st.session_state.weight_classes = [
        ("Fliegengewicht", 51.0), ("Bantamgewicht", 54.0), ("Federgewicht", 57.0), ("Leichtgewicht", 60.0),
        ("Halbweltergewicht", 63.5), ("Weltergewicht", 67.0), ("Halbmittelgewicht", 71.0), ("Mittelgewicht", 75.0),
        ("Halbschwergewicht", 81.0), ("Cruisergewicht", 86.0), ("Schwergewicht", 91.0)
    ]

if "current_matches" not in st.session_state: st.session_state.current_matches = pd.DataFrame()
if "current_unmatched" not in st.session_state: st.session_state.current_unmatched = pd.DataFrame()

with st.sidebar:
    st.header("⚙️ Matching-Regeln")
    same_gender_only = st.checkbox("Nur gleiches Geschlecht matchen", value=True)
    strict_discipline_matching = st.checkbox("Strikte Disziplinen-Trennung", value=True)
    strict_class_matching = st.checkbox("Strikte Klassen-Trennung (Erfahrung)", value=False)
    strict_weight_class = st.checkbox("Strikte Gewichtsklassen-Trennung", value=False)
    
    st.markdown("---")
    max_weight_diff = st.number_input("Max. Gewichtsdifferenz in kg", min_value=0.1, value=5.0)
    max_age_diff = st.number_input("Max. Altersdifferenz", min_value=1, value=8)
    max_fight_diff = st.number_input("Max. Kampfdifferenz", min_value=0, value=5)
    min_score = st.slider("Mindest-Score", min_value=0, max_value=100, value=50)

    st.markdown("---")
    st.subheader("📉 Strafen-Multiplikator")
    w_weight = st.slider("Gewichtung: Gewicht", 0.0, 5.0, 3.0, 0.5)
    w_age = st.slider("Gewichtung: Alter", 0.0, 5.0, 1.5, 0.5)
    w_fights = st.slider("Gewichtung: Anzahl Kämpfe", 0.0, 5.0, 2.0, 0.5)

    st.download_button("Excel-Vorlage herunterladen", data=template_excel(), file_name="muay_thai_k1_vorlage.xlsx")

settings = MatchSettings(
    max_weight_diff=float(max_weight_diff), max_age_diff=int(max_age_diff), max_fight_diff=int(max_fight_diff), max_win_diff=5, max_loss_diff=5,
    min_score=float(min_score), same_gender_only=bool(same_gender_only), strict_class_matching=bool(strict_class_matching),
    strict_weight_class=bool(strict_weight_class), strict_discipline_matching=bool(strict_discipline_matching),
    weights=Weights(w_weight, w_age, w_fights, 1.0, 1.0, 1.0),
)

# BACKUP & WIEDERHERSTELLUNG
st.subheader("📂 Backup & Bearbeitungsstand laden")
saved_state_file = st.file_uploader("Gesicherten Bearbeitungsstand laden (.json)", type=["json"])

if saved_state_file is not None:
    try:
        state_data = json.load(saved_state_file)
        st.session_state.current_matches = pd.DataFrame(state_data["matches"])
        st.session_state.current_unmatched = pd.DataFrame(state_data["unmatched"])
        st.success("Bearbeitungsstand geladen!")
    except Exception as e:
        st.error(f"Fehler: {e}")

st.markdown("---")

# EXCEL NEU IMPORTIEREN
uploaded_file = st.file_uploader("Meldeliste hochladen (.xlsx)", type=["xlsx"])
if uploaded_file is not None and saved_state_file is None:
    if st.button("🚀 Neue Paarungen aus Meldeliste generieren"):
        try:
            raw_df = pd.read_excel(uploaded_file)
            fighters, validation_errors = clean_fighters(raw_df, st.session_state.weight_classes)
            if validation_errors:
                st.error("Fehler in der Excel-Datei!")
                st.stop()
            matches_df, unmatched_df = build_matches(fighters, settings)
            st.session_state.current_matches = matches_df
            st.session_state.current_unmatched = unmatched_df
            st.success("Paarungen generiert!")
        except Exception as exc:
            st.error(f"Fehler: {exc}")

# ECHTES DRAG AND DROP INTERFACE
if not st.session_state.current_matches.empty:
    st.markdown("---")
    st.header("🔀 Kampfreihenfolge per Drag-and-Drop festlegen")
    st.info("Klicke eine Paarung an und ziehe sie mit der Maus nach oben oder unten, um die Reihenfolge fehlerfrei zu ändern.")

    # Bereite die Liste der Beschriftungen für die Drag-Karten vor
    current_list = st.session_state.current_matches["id"].tolist()
    
    # Hier passiert die Magie der Drag-and-Drop-Komponente
    sorted_id_list = sort_items(current_list, direction="vertical", key="drag_drop_matches")

    # Wenn der Benutzer umsortiert hat, passe das Dataframe an und aktualisiere die Kampfnummern
    if sorted_id_list != current_list:
        # Sortiere das Dataframe anhand der neuen Drag-Reihenfolge
        st.session_state.current_matches['id'] = pd.Categorical(st.session_state.current_matches['id'], categories=sorted_id_list, ordered=True)
        st.session_state.current_matches = st.session_state.current_matches.sort_values('id').reset_index(drop=True)
        # Vergib die Kampfnummern lückenlos neu von 1 bis X
        st.session_state.current_matches["Kampf Nr."] = range(1, len(st.session_state.current_matches) + 1)
        st.rerender()

    # Übersichtliche Tabellen-Vorschau der aktuell gesetzten Reihenfolge
    st.subheader("Aktuelle Kampfliste Übersicht")
    st.dataframe(
        st.session_state.current_matches.drop(columns=["id"]), 
        use_container_width=True, 
        hide_index=True
    )

    # RE-EXPORT ALS JSON
    st.subheader("💾 Aktuellen Arbeitsstand sichern")
    state_json = json.dumps({
        "matches": st.session_state.current_matches.to_dict(orient="records"),
        "unmatched": st.session_state.current_unmatched.to_dict(orient="records")
    }, indent=2)
    
    st.download_button(
        "💾 Aktuellen Bearbeitungsstand sichern (.json)",
        data=state_json,
        file_name="matchmaker_bearbeitungsstand.json",
        mime="application/json"
    )

    # FINALE EXCEL
    st.subheader("📊 Finaler Export")
    export_bytes = export_to_excel(st.session_state.current_matches.drop(columns=["id"]), st.session_state.current_unmatched)
    st.download_button(
        "🖨️ Finale Kampfliste als Excel herunterladen",
        data=export_bytes,
        file_name="finale_kampfliste_turnier.xlsx"
    )

if not st.session_state.current_unmatched.empty:
    with st.expander("Verbleibende ungematchte Kämpfer anzeigen"):
        st.dataframe(st.session_state.current_unmatched, use_container_width=True, hide_index=True)

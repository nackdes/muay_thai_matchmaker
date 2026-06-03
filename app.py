import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import streamlit as st

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
    "name": "Name",
    "kaempfer": "Name",
    "kämpfer": "Name",
    "fighter": "Name",
    "vorname nachname": "Name",
    "verein": "Verein",
    "club": "Verein",
    "gym": "Verein",
    "team": "Verein",
    "geschlecht": "Geschlecht",
    "gender": "Geschlecht",
    "sex": "Geschlecht",
    "alter": "Alter",
    "age": "Alter",
    "gewicht": "Gewicht",
    "gewicht kg": "Gewicht",
    "gewicht (kg)": "Gewicht",
    "kg": "Gewicht",
    "disziplin": "Disziplin",
    "discipline": "Disziplin",
    "regelwerk": "Disziplin",
    "art": "Disziplin",
    "kaempfe": "Kämpfe",
    "kämpfe": "Kämpfe",
    "anzahl kaempfe": "Kämpfe",
    "anzahl kämpfe": "Kämpfe",
    "anzahl an kämpfen": "Kämpfe",
    "fights": "Kämpfe",
    "total fights": "Kämpfe",
    "siege": "Siege",
    "gewonnen": "Siege",
    "gewonnene kaempfe": "Siege",
    "gewonnene kämpfe": "Siege",
    "wins": "Siege",
    "niederlagen": "Niederlagen",
    "verloren": "Niederlagen",
    "verlorene kaempfe": "Niederlagen",
    "verlorene kämpfe": "Niederlagen",
    "losses": "Niederlagen",
}


@dataclass
class Weights:
    weight: float
    age: float
    fights: float
    wins: float
    losses: float
    winrate: float


@dataclass
class MatchSettings:
    max_weight_diff: float
    max_age_diff: int
    max_fight_diff: int
    max_win_diff: int
    max_loss_diff: int
    min_score: float
    same_gender_only: bool
    strict_class_matching: bool
    strict_weight_class: bool
    strict_discipline_matching: bool
    weights: Weights


def normalize_header(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[\n\r\t_\-/]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_gender(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    if text in {"m", "maennlich", "mann", "male", "herr", "junge"}:
        return "männlich"
    if text in {"w", "weiblich", "frau", "female", "dame", "maedchen"}:
        return "weiblich"
    if text in {"d", "divers", "nonbinary", "non binary", "nb"}:
        return "divers"
    return str(value).strip()


def normalize_discipline(value: object) -> str:
    text = str(value).strip().lower().replace(" ", "").replace("-", "")
    if "k1" in text or "kick" in text:
        return "K-1"
    if "muay" in text or "thai" in text or "boxen" in text:
        return "Muay Thai"
    return str(value).strip()


def normalize_club(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def get_experience_class(fights: int) -> str:
    if fights == 0:
        return "Newcomer (0)"
    elif fights <= 5:
        return "D-Klasse (1-5)"
    elif fights <= 15:
        return "C-Klasse (6-15)"
    elif fights <= 25:
        return "B-Klasse (16-25)"
    else:
        return "A-Klasse (>25)"


def get_weight_class(weight: float, weight_classes: List[Tuple[str, float]]) -> str:
    sorted_classes = sorted(weight_classes, key=lambda x: x[1])
    for name, limit in sorted_classes:
        if weight <= limit:
            return f"{name} (-{limit}kg)"
    return f"Superschwergewicht (> {sorted_classes[-1][1]}kg)" if sorted_classes else "Unbekannt"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    used_targets = set()

    for col in df.columns:
        key = normalize_header(col)
        key_ascii = key.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
        target = COLUMN_ALIASES.get(key) or COLUMN_ALIASES.get(key_ascii)
        if target and target not in used_targets:
            rename_map[col] = target
            used_targets.add(target)

    return df.rename(columns=rename_map)


def clean_fighters(raw_df: pd.DataFrame, weight_classes: List[Tuple[str, float]]) -> Tuple[pd.DataFrame, List[str]]:
    df = normalize_columns(raw_df).copy()
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError("Diese Pflichtspalten fehlen: " + ", ".join(missing))

    df = df[REQUIRED_COLUMNS].copy()
    df = df.dropna(how="all")

    for col in ["Name", "Verein", "Geschlecht", "Disziplin"]:
        df[col] = df[col].astype(str).str.strip()

    for col in ["Alter", "Gewicht", "Kämpfe", "Siege", "Niederlagen"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Geschlecht_norm"] = df["Geschlecht"].apply(normalize_gender)
    df["Disziplin_norm"] = df["Disziplin"].apply(normalize_discipline)
    df["Verein_norm"] = df["Verein"].apply(normalize_club)
    df["Winrate"] = df.apply(lambda r: float(r["Siege"]) / float(r["Kämpfe"]) if r["Kämpfe"] and r["Kämpfe"] > 0 else 0.0, axis=1)

    errors: List[str] = []
    for idx, row in df.iterrows():
        line = idx + 2
        if not row["Name"] or row["Name"].lower() == "nan":
            errors.append(f"Zeile {line}: Name fehlt.")
        if not row["Verein"] or row["Verein"].lower() == "nan":
            errors.append(f"Zeile {line}: Verein fehlt.")
        if not row["Geschlecht"] or row["Geschlecht"].lower() == "nan":
            errors.append(f"Zeile {line}: Geschlecht fehlt.")
        if not row["Disziplin"] or row["Disziplin"].lower() == "nan":
            errors.append(f"Zeile {line}: Disziplin fehlt.")
        for col in ["Alter", "Gewicht", "Kämpfe", "Siege", "Niederlagen"]:
            if pd.isna(row[col]):
                errors.append(f"Zeile {line}: {col} ist keine Zahl.")
            elif float(row[col]) < 0:
                errors.append(f"Zeile {line}: {col} darf nicht negativ sein.")
        if not pd.isna(row["Kämpfe"]) and not pd.isna(row["Siege"]) and not pd.isna(row["Niederlagen"]):
            if float(row["Siege"]) + float(row["Niederlagen"]) > float(row["Kämpfe"]):
                errors.append(f"Zeile {line}: Siege + Niederlagen ist größer als Kämpfe.")

    if errors:
        return df, errors

    df["Alter"] = df["Alter"].round().astype(int)
    df["Kämpfe"] = df["Kämpfe"].round().astype(int)
    df["Siege"] = df["Siege"].round().astype(int)
    df["Niederlagen"] = df["Niederlagen"].round().astype(int)
    df["Gewicht"] = df["Gewicht"].astype(float)
    df["Klasse"] = df["Kämpfe"].apply(get_experience_class)
    df["Gewichtsklasse"] = df["Gewicht"].apply(lambda w: get_weight_class(w, weight_classes))
    return df.reset_index(drop=True), []


def score_pair(a: pd.Series, b: pd.Series, settings: MatchSettings) -> Optional[Dict[str, float]]:
    if a["Verein_norm"] == b["Verein_norm"]:
        return None
    if settings.same_gender_only and a["Geschlecht_norm"] != b["Geschlecht_norm"]:
        return None
    if settings.strict_class_matching and a["Klasse"] != b["Klasse"]:
        return None
    if settings.strict_weight_class and a["Gewichtsklasse"] != b["Gewichtsklasse"]:
        return None
    if settings.strict_discipline_matching and a["Disziplin_norm"] != b["Disziplin_norm"]:
        return None

    weight_diff = abs(float(a["Gewicht"]) - float(b["Gewicht"]))
    age_diff = abs(int(a["Alter"]) - int(b["Alter"]))
    fight_diff = abs(int(a["Kämpfe"]) - int(b["Kämpfe"]))
    win_diff = abs(int(a["Siege"]) - int(b["Siege"]))
    loss_diff = abs(int(a["Niederlagen"]) - int(b["Niederlagen"]))
    winrate_diff = abs(float(a["Winrate"]) - float(b["Winrate"]))

    if weight_diff > settings.max_weight_diff:
        return None
    if age_diff > settings.max_age_diff:
        return None
    if fight_diff > settings.max_fight_diff:
        return None
    if win_diff > settings.max_win_diff:
        return None
    if loss_diff > settings.max_loss_diff:
        return None

    weights_sum = (
        settings.weights.weight
        + settings.weights.age
        + settings.weights.fights
        + settings.weights.wins
        + settings.weights.losses
        + settings.weights.winrate
    )
    if weights_sum <= 0:
        weights_sum = 1.0

    penalty = 0.0
    penalty += settings.weights.weight * min(weight_diff / max(settings.max_weight_diff, 0.1), 1.0)
    penalty += settings.weights.age * min(age_diff / max(settings.max_age_diff, 1), 1.0)
    penalty += settings.weights.fights * min(fight_diff / max(settings.max_fight_diff, 1), 1.0)
    penalty += settings.weights.wins * min(win_diff / max(settings.max_win_diff, 1), 1.0)
    penalty += settings.weights.losses * min(loss_diff / max(settings.max_loss_diff, 1), 1.0)
    penalty += settings.weights.winrate * min(winrate_diff, 1.0)

    score = max(0.0, 100.0 * (1.0 - penalty / weights_sum))
    
    # Kleiner Abzug falls Disziplinen gemischt werden (wenn nicht strikt getrennt)
    if a["Disziplin_norm"] != b["Disziplin_norm"]:
        score -= 15.0
        score = max(0.0, score)

    if score < settings.min_score:
        return None

    return {
        "Score": round(score, 1),
        "Gewichtsdifferenz": round(weight_diff, 2),
        "Altersdifferenz": age_diff,
        "Kampfdifferenz": fight_diff,
        "Siege-Differenz": win_diff,
        "Niederlagen-Differenz": loss_diff,
        "Winrate-Differenz": round(winrate_diff, 3),
    }


def build_matches(fighters: pd.DataFrame, settings: MatchSettings) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    graph = nx.Graph()
    for idx in fighters.index:
        graph.add_node(idx)

    candidate_rows = []
    for i in fighters.index:
        for j in fighters.index:
            if j <= i:
                continue
            details = score_pair(fighters.loc[i], fighters.loc[j], settings)
            if details is None:
                continue
            weight_for_graph = int(details["Score"] * 10000) - int(details["Gewichtsdifferenz"] * 10) - details["Kampfdifferenz"]
            graph.add_edge(i, j, weight=weight_for_graph, details=details)
            candidate_rows.append(make_output_row(fighters.loc[i], fighters.loc[j], details))

    matching = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True, weight="weight")

    match_rows = []
    matched_indices = set()
    for i, j in sorted(matching, key=lambda pair: -graph.edges[pair]["details"]["Score"]):
        details = graph.edges[i, j]["details"]
        match_rows.append(make_output_row(fighters.loc[i], fighters.loc[j], details))
        matched_indices.update([i, j])

    matches_df = pd.DataFrame(match_rows)
    candidates_df = pd.DataFrame(candidate_rows).sort_values(by="Score", ascending=False) if candidate_rows else pd.DataFrame()

    unmatched_rows = []
    for idx, fighter in fighters.iterrows():
        if idx in matched_indices:
            continue
        unmatched_rows.append(
            {
                "ID": idx,
                "Name": fighter["Name"],
                "Verein": fighter["Verein"],
                "Geschlecht": fighter["Geschlecht_norm"],
                "Alter": fighter["Alter"],
                "Gewicht": fighter["Gewicht"],
                "Gewichtsklasse": fighter["Gewichtsklasse"],
                "Disziplin": fighter["Disziplin_norm"],
                "Kämpfe": fighter["Kämpfe"],
                "Klasse": fighter["Klasse"],
                "Bilanz": f'{fighter["Siege"]}-{fighter["Niederlagen"]}',
            }
        )
    unmatched_df = pd.DataFrame(unmatched_rows)
    return matches_df, unmatched_df, candidates_df


def make_output_row(a: pd.Series, b: pd.Series, details: Dict[str, float]) -> Dict[str, object]:
    # Bestimme Disziplin-Label für die Paarung
    match_discipline = a["Disziplin_norm"] if a["Disziplin_norm"] == b["Disziplin_norm"] else f"{a['Disziplin_norm']} vs {b['Disziplin_norm']}"
    return {
        "Kampf-Disziplin": match_discipline,
        "Kämpfer A": a["Name"],
        "Verein A": a["Verein"],
        "Geschlecht A": a["Geschlecht_norm"],
        "Alter A": a["Alter"],
        "Gewicht A": a["Gewicht"],
        "Gewichtsklasse A": a["Gewichtsklasse"],
        "Kämpfe A": a["Kämpfe"],
        "Klasse A": a["Klasse"],
        "Bilanz A": f'{a["Siege"]}-{a["Niederlagen"]}',
        "Kämpfer B": b["Name"],
        "Verein B": b["Verein"],
        "Geschlecht B": b["Geschlecht_norm"],
        "Alter B": b["Alter"],
        "Gewicht B": b["Gewicht"],
        "Gewichtsklasse B": b["Gewichtsklasse"],
        "Kämpfe B": b["Kämpfe"],
        "Klasse B": b["Klasse"],
        "Bilanz B": f'{b["Siege"]}-{b["Niederlagen"]}',
        **details,
    }


def export_to_excel(matches: pd.DataFrame, unmatched: pd.DataFrame, candidates: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        matches.to_excel(writer, sheet_name="Matches", index=False)
        unmatched.to_excel(writer, sheet_name="Nicht gematcht", index=False)
        candidates.head(250).to_excel(writer, sheet_name="Top Kandidaten", index=False)
    return output.getvalue()


def template_excel() -> bytes:
    sample = pd.DataFrame(
        [
            ["Kämpfer 1", "Gym Alpha", "männlich", 25, 70.0, "Muay Thai", 3, 2, 1],
            ["Kämpfer 2", "Gym Beta", "männlich", 27, 71.2, "Muay Thai", 4, 2, 2],
            ["Kämpfer 3", "Gym Gamma", "weiblich", 19, 58.5, "K-1", 1, 1, 0],
            ["Kämpfer 4", "Gym Delta", "weiblich", 21, 59.0, "K-1", 2, 1, 1],
        ],
        columns=REQUIRED_COLUMNS,
    )
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        sample.to_excel(writer, sheet_name="Kämpfer", index=False)
    return output.getvalue()


# Streamlit Seiten-Konfiguration
st.set_page_config(page_title="Muay Thai & K1 Matchmaker", page_icon="🥊", layout="wide")
st.title("🥊 Muay Thai & K1 Matchmaker Pro")
st.caption("Importiere eine Excel-Datei und finde optimale, nicht überlappende Kampfpaarungen getrennt nach Regelwerk.")

# Initialisiere Standard-Gewichtsklassen im Session State
if "weight_classes" not in st.session_state:
    st.session_state.weight_classes = [
        ("Fliegengewicht", 51.0),
        ("Bantamgewicht", 54.0),
        ("Federgewicht", 57.0),
        ("Leichtgewicht", 60.0),
        ("Halbweltergewicht", 63.5),
        ("Weltergewicht", 67.0),
        ("Halbmittelgewicht", 71.0),
        ("Mittelgewicht", 75.0),
        ("Halbschwergewicht", 81.0),
        ("Cruisergewicht", 86.0),
        ("Schwergewicht", 91.0)
    ]

with st.sidebar:
    st.header("⚙️ Matching-Regeln")
    same_gender_only = st.checkbox(
        "Nur gleiches Geschlecht matchen", 
        value=True,
        help="Wenn aktiv, können Personen mit unterschiedlichen Geschlechtseinträgen niemals gegeneinander gelost werden."
    )
    strict_discipline_matching = st.checkbox(
        "Strikte Disziplinen-Trennung",
        value=True,
        help="Erzwingt, dass Muay Thai Kämpfer nur gegen Muay Thai Kämpfer und K1 Kämpfer nur gegen K1 Kämpfer antreten. Wenn deaktiviert, sind stilübergreifende Kämpfe mit Punktabzug erlaubt."
    )
    strict_class_matching = st.checkbox(
        "Strikte Klassen-Trennung (Erfahrung)", 
        value=False,
        help="Teilt Kämpfer anhand ihrer Kämpfe in Leistungsklassen ein (z.B. Newcomer, C-Klasse). Bei Aktivierung wird klassenübergreifendes Matchen komplett blockiert."
    )
    strict_weight_class = st.checkbox(
        "Strikte Gewichtsklassen-Trennung", 
        value=False,
        help="Erzwingt, dass Kämpfer NUR innerhalb der exakt selben Gewichtsklasse (z.B. Weltergewicht) gepaart werden. Wenn deaktiviert, entscheidet die kg-Differenz."
    )
    
    st.markdown("---")
    max_weight_diff = st.number_input(
        "Max. Gewichtsdifferenz in kg", 
        min_value=0.1, max_value=30.0, value=5.0, step=0.5,
        help="Die absolute Obergrenze, wie viele Kilo zwei Gegner auseinander sein dürfen."
    )
    max_age_diff = st.number_input(
        "Max. Altersdifferenz", 
        min_value=1, max_value=60, value=8, step=1,
        help="Der maximale Altersunterschied in Jahren, der zwischen zwei Kämpfern liegen darf."
    )
    max_fight_diff = st.number_input(
        "Max. Kampfdifferenz", 
        min_value=0, max_value=100, value=5, step=1,
        help="Die maximale Differenz in der Anzahl absolvierter Kämpfe zwischen beiden Kontrahenten."
    )
    max_win_diff = st.number_input("Max. Siege-Differenz", min_value=0, max_value=100, value=5, step=1)
    max_loss_diff = st.number_input("Max. Niederlagen-Differenz", min_value=0, max_value=100, value=5, step=1)
    
    min_score = st.slider(
        "Mindest-Score", 
        min_value=0, max_value=100, value=50, step=1,
        help="Der Qualitätsfilter für das 'Gesamtpaket'. Jedes Paar startet bei 100 Punkten. Unterschiede geben Abzüge. Fällt ein Paar unter diesen Mindestwert, wird der Kampf verboten."
    )

    st.header("⚖️ Gewichtsklassen verwalten")
    with st.form("new_weight_class_form", clear_on_submit=True):
        new_name = st.text_input("Name der Klasse (z.B. Welter)")
        new_limit = st.number_input("Maximalgewicht (kg)", min_value=1.0, max_value=200.0, value=70.0, step=0.5)
        submitted = st.form_submit_button("Hinzufügen")
        if submitted and new_name:
            st.session_state.weight_classes.append((new_name, float(new_limit)))
            st.rerender()

    st.subheader("Aktuelle Klassen (Obergrenze):")
    sorted_classes = sorted(st.session_state.weight_classes, key=lambda x: x[1])
    
    to_delete = None
    for idx, (name, limit) in enumerate(sorted_classes):
        col_c1, col_c2 = st.columns([3, 1])
        col_c1.write(f"**{name}**: -{limit} kg")
        if col_c2.button("❌", key=f"del_{idx}"):
            to_delete = (name, limit)
            
    if to_delete:
        st.session_state.weight_classes.remove(to_delete)
        st.rerender()

    st.markdown("---")
    st.subheader("📉 Strafen-Multiplikator")
    w_weight = st.slider("Gewichtung: Gewicht", 0.0, 5.0, 3.0, 0.5)
    w_age = st.slider("Gewichtung: Alter", 0.0, 5.0, 1.5, 0.5)
    w_fights = st.slider("Gewichtung: Anzahl Kämpfe", 0.0, 5.0, 2.0, 0.5)
    w_wins = st.slider("Gewichtung: Siege", 0.0, 5.0, 1.0, 0.5)
    w_losses = st.slider("Gewichtung: Niederlagen", 0.0, 5.0, 1.0, 0.5)
    w_winrate = st.slider("Gewichtung: Siegquote", 0.0, 5.0, 1.0, 0.5)

    st.download_button(
        "Excel-Vorlage herunterladen",
        data=template_excel(),
        file_name="muay_thai_k1_vorlage.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

settings = MatchSettings(
    max_weight_diff=float(max_weight_diff),
    max_age_diff=int(max_age_diff),
    max_fight_diff=int(max_fight_diff),
    max_win_diff=int(max_win_diff),
    max_loss_diff=int(max_loss_diff),
    min_score=float(min_score),
    same_gender_only=bool(same_gender_only),
    strict_class_matching=bool(strict_class_matching),
    strict_weight_class=bool(strict_weight_class),
    strict_discipline_matching=bool(strict_discipline_matching),
    weights=Weights(w_weight, w_age, w_fights, w_wins, w_losses, w_winrate),
)

uploaded_file = st.file_uploader("Excel-Datei hochladen (.xlsx)", type=["xlsx"])

st.info(f"Pflichtspalten in der Excel: {', '.join(REQUIRED_COLUMNS)}")

if uploaded_file is not None:
    try:
        raw_df = pd.read_excel(uploaded_file)
        fighters, validation_errors = clean_fighters(raw_df, st.session_state.weight_classes)
        if validation_errors:
            st.error("Bitte korrigiere die Importdatei.")
            for err in validation_errors[:50]:
                st.write("- " + err)
            if len(validation_errors) > 50:
                st.write(f"… und {len(validation_errors) - 50} weitere Fehler.")
            st.stop()

        st.subheader("Importierte Kämpfer")
        st.dataframe(fighters[["Name", "Verein", "Geschlecht", "Alter", "Gewicht", "Gewichtsklasse", "Disziplin_norm", "Kämpfe", "Klasse"]], use_container_width=True, hide_index=True)

        matches_df, unmatched_df, candidates_df = build_matches(fighters, settings)

        col1, col2, col3 = st.columns(3)
        col1.metric("Kämpfer Gesamt", len(fighters))
        col2.metric("Automatische Matches", len(matches_df))
        col3.metric("Noch ungematcht", len(unmatched_df))

        st.subheader("Optimale Paarungen (Algorithmus)")
        if matches_df.empty:
            st.warning("Keine automatischen Paarungen gefunden. Passe die Filter an.")
        else:
            st.dataframe(matches_df, use_container_width=True, hide_index=True)

        # Manueller Matchmaker für ungematchte Kämpfer
        st.subheader("🛠️ Manueller Matchmaker für Nachzügler")
        if not unmatched_df.empty and len(unmatched_df) >= 2:
            st.write("Verwende dieses Tool, um Kämpfer manuell zu paaren, bei denen der Algorithmus restriktiv war.")
            
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                f1_options = {f"{row['Name']} ({row['Verein']} - {row['Disziplin']} - {row['Gewicht']}kg)": row['ID'] for _, row in unmatched_df.iterrows()}
                fighter_1_label = st.selectbox("Wähle Kämpfer A:", options=list(f1_options.keys()), key="man_f1")
                f1_id = f1_options[fighter_1_label]
                
            with col_m2:
                f2_options = {f"{row['Name']} ({row['Verein']} - {row['Disziplin']} - {row['Gewicht']}kg)": row['ID'] for _, row in unmatched_df.iterrows() if row['ID'] != f1_id}
                fighter_2_label = st.selectbox("Wähle Kämpfer B:", options=list(f2_options.keys()), key="man_f2")
                
            if st.button("Manuelles Match erzwingen und hinzufügen"):
                f2_id = f2_options[fighter_2_label]
                
                fa_series = fighters.loc[f1_id]
                fb_series = fighters.loc[f2_id]
                
                man_details = {
                    "Score": 100.0,
                    "Gewichtsdifferenz": round(abs(fa_series["Gewicht"] - fb_series["Gewicht"]), 2),
                    "Altersdifferenz": abs(fa_series["Alter"] - fb_series["Alter"]),
                    "Kampfdifferenz": abs(fa_series["Kämpfe"] - fb_series["Kämpfe"]),
                    "Siege-Differenz": abs(fa_series["Siege"] - fb_series["Siege"]),
                    "Niederlagen-Differenz": abs(fa_series["Niederlagen"] - fb_series["Niederlagen"]),
                    "Winrate-Differenz": round(abs(fa_series["Winrate"] - fb_series["Winrate"]), 3)
                }
                
                new_row = pd.DataFrame([make_output_row(fa_series, fb_series, man_details)])
                
                if "manual_matches" not in st.session_state:
                    st.session_state.manual_matches = pd.DataFrame()
                
                st.session_state.manual_matches = pd.concat([st.session_state.manual_matches, new_row], ignore_index=True)
                st.success(f"Match zwischen {fa_series['Name']} und {fb_series['Name']} wurde hinzugefügt!")
                st.rerender()
        else:
            st.write("Nicht genug ungematchte Kämpfer für eine manuelle Auswahl.")

        if "manual_matches" in st.session_state and not st.session_state.manual_matches.empty:
            st.subheader("➕ Manuell hinzugefügte Paarungen")
            st.dataframe(st.session_state.manual_matches, use_container_width=True, hide_index=True)
            if st.button("Alle manuellen Matches zurücksetzen"):
                st.session_state.manual_matches = pd.DataFrame()
                st.rerender()
            
            final_matches_df = pd.concat([matches_df, st.session_state.manual_matches], ignore_index=True)
            man_names = set(st.session_state.manual_matches["Kämpfer A"]).union(set(st.session_state.manual_matches["Kämpfer B"]))
            final_unmatched_df = unmatched_df[~unmatched_df["Name"].isin(man_names)]
        else:
            final_matches_df = matches_df
            final_unmatched_df = unmatched_df

        with st.expander("Verbleibende ungematchte Kämpfer"):
            if final_unmatched_df.empty:
                st.success("Alle Kämpfer wurden erfolgreich untergebracht!")
            else:
                st.dataframe(final_unmatched_df.drop(columns=["ID"]), use_container_width=True, hide_index=True)

        with st.expander("Alle mathematisch gültigen Match-Kandidaten (Top 250)"):
            if candidates_df.empty:
                st.write("Keine gültigen Kandidatenpaare im Toleranzbereich gefunden.")
            else:
                st.dataframe(candidates_df.head(250), use_container_width=True, hide_index=True)

        export_bytes = export_to_excel(final_matches_df, final_unmatched_df, candidates_df)
        st.download_button(
            "Finale Kampfliste als Excel herunterladen",
            data=export_bytes,
            file_name="muay_thai_turnier_matches.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as exc:
        st.error(f"Import oder Matching fehlgeschlagen: {exc}")
else:
    st.write("Lade eine Excel-Datei hoch oder nutze zuerst die Vorlage aus der Seitenleiste.")

import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import streamlit as st


REQUIRED_COLUMNS = [
    "Name",
    "Verein",
    "Geschlecht",
    "Gewicht",
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
    "gewicht": "Gewicht",
    "gewicht kg": "Gewicht",
    "gewicht (kg)": "Gewicht",
    "kg": "Gewicht",
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
    fights: float
    wins: float
    losses: float
    winrate: float


@dataclass
class MatchSettings:
    max_weight_diff: float
    max_fight_diff: int
    max_win_diff: int
    max_loss_diff: int
    min_score: float
    same_gender_only: bool
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


def normalize_club(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


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


def clean_fighters(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = normalize_columns(raw_df).copy()
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError("Diese Pflichtspalten fehlen: " + ", ".join(missing))

    df = df[REQUIRED_COLUMNS].copy()
    df = df.dropna(how="all")

    for col in ["Name", "Verein", "Geschlecht"]:
        df[col] = df[col].astype(str).str.strip()

    for col in ["Gewicht", "Kämpfe", "Siege", "Niederlagen"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Geschlecht_norm"] = df["Geschlecht"].apply(normalize_gender)
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
        for col in ["Gewicht", "Kämpfe", "Siege", "Niederlagen"]:
            if pd.isna(row[col]):
                errors.append(f"Zeile {line}: {col} ist keine Zahl.")
            elif float(row[col]) < 0:
                errors.append(f"Zeile {line}: {col} darf nicht negativ sein.")
        if not pd.isna(row["Kämpfe"]) and not pd.isna(row["Siege"]) and not pd.isna(row["Niederlagen"]):
            if float(row["Siege"]) + float(row["Niederlagen"]) > float(row["Kämpfe"]):
                errors.append(f"Zeile {line}: Siege + Niederlagen ist größer als Kämpfe.")

    if errors:
        return df, errors

    df["Kämpfe"] = df["Kämpfe"].round().astype(int)
    df["Siege"] = df["Siege"].round().astype(int)
    df["Niederlagen"] = df["Niederlagen"].round().astype(int)
    df["Gewicht"] = df["Gewicht"].astype(float)
    return df.reset_index(drop=True), []


def score_pair(a: pd.Series, b: pd.Series, settings: MatchSettings) -> Optional[Dict[str, float]]:
    if a["Verein_norm"] == b["Verein_norm"]:
        return None
    if settings.same_gender_only and a["Geschlecht_norm"] != b["Geschlecht_norm"]:
        return None

    weight_diff = abs(float(a["Gewicht"]) - float(b["Gewicht"]))
    fight_diff = abs(int(a["Kämpfe"]) - int(b["Kämpfe"]))
    win_diff = abs(int(a["Siege"]) - int(b["Siege"]))
    loss_diff = abs(int(a["Niederlagen"]) - int(b["Niederlagen"]))
    winrate_diff = abs(float(a["Winrate"]) - float(b["Winrate"]))

    if weight_diff > settings.max_weight_diff:
        return None
    if fight_diff > settings.max_fight_diff:
        return None
    if win_diff > settings.max_win_diff:
        return None
    if loss_diff > settings.max_loss_diff:
        return None

    weights_sum = (
        settings.weights.weight
        + settings.weights.fights
        + settings.weights.wins
        + settings.weights.losses
        + settings.weights.winrate
    )
    if weights_sum <= 0:
        weights_sum = 1.0

    penalty = 0.0
    penalty += settings.weights.weight * min(weight_diff / max(settings.max_weight_diff, 0.1), 1.0)
    penalty += settings.weights.fights * min(fight_diff / max(settings.max_fight_diff, 1), 1.0)
    penalty += settings.weights.wins * min(win_diff / max(settings.max_win_diff, 1), 1.0)
    penalty += settings.weights.losses * min(loss_diff / max(settings.max_loss_diff, 1), 1.0)
    penalty += settings.weights.winrate * min(winrate_diff, 1.0)

    score = max(0.0, 100.0 * (1.0 - penalty / weights_sum))
    if score < settings.min_score:
        return None

    return {
        "Score": round(score, 1),
        "Gewichtsdifferenz": round(weight_diff, 2),
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
                "Name": fighter["Name"],
                "Verein": fighter["Verein"],
                "Geschlecht": fighter["Geschlecht_norm"],
                "Gewicht": fighter["Gewicht"],
                "Kämpfe": fighter["Kämpfe"],
                "Siege": fighter["Siege"],
                "Niederlagen": fighter["Niederlagen"],
            }
        )
    unmatched_df = pd.DataFrame(unmatched_rows)
    return matches_df, unmatched_df, candidates_df


def make_output_row(a: pd.Series, b: pd.Series, details: Dict[str, float]) -> Dict[str, object]:
    return {
        "Kämpfer A": a["Name"],
        "Verein A": a["Verein"],
        "Geschlecht A": a["Geschlecht_norm"],
        "Gewicht A": a["Gewicht"],
        "Kämpfe A": a["Kämpfe"],
        "Bilanz A": f'{a["Siege"]}-{a["Niederlagen"]}',
        "Kämpfer B": b["Name"],
        "Verein B": b["Verein"],
        "Geschlecht B": b["Geschlecht_norm"],
        "Gewicht B": b["Gewicht"],
        "Kämpfe B": b["Kämpfe"],
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
            ["Kämpfer 1", "Gym Alpha", "männlich", 70.0, 3, 2, 1],
            ["Kämpfer 2", "Gym Beta", "männlich", 71.2, 4, 2, 2],
            ["Kämpfer 3", "Gym Gamma", "weiblich", 58.5, 1, 1, 0],
            ["Kämpfer 4", "Gym Delta", "weiblich", 59.0, 2, 1, 1],
        ],
        columns=REQUIRED_COLUMNS,
    )
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        sample.to_excel(writer, sheet_name="Kämpfer", index=False)
    return output.getvalue()


st.set_page_config(page_title="Muay Thai Matchmaker", page_icon="🥊", layout="wide")
st.title("🥊 Muay Thai Matchmaker")
st.caption("Importiere eine Excel-Datei und finde optimale, nicht überlappende Kampfpaarungen. Gleicher Verein wird immer ausgeschlossen.")

with st.sidebar:
    st.header("Matching-Regeln")
    same_gender_only = st.checkbox("Nur gleiches Geschlecht matchen", value=True)
    max_weight_diff = st.number_input("Max. Gewichtsdifferenz in kg", min_value=0.1, max_value=30.0, value=5.0, step=0.5)
    max_fight_diff = st.number_input("Max. Kampfdifferenz", min_value=0, max_value=100, value=5, step=1)
    max_win_diff = st.number_input("Max. Siege-Differenz", min_value=0, max_value=100, value=5, step=1)
    max_loss_diff = st.number_input("Max. Niederlagen-Differenz", min_value=0, max_value=100, value=5, step=1)
    min_score = st.slider("Mindest-Score", min_value=0, max_value=100, value=55, step=1)

    st.subheader("Gewichtung")
    w_weight = st.slider("Gewicht", 0.0, 5.0, 3.0, 0.5)
    w_fights = st.slider("Anzahl Kämpfe", 0.0, 5.0, 2.0, 0.5)
    w_wins = st.slider("Siege", 0.0, 5.0, 1.5, 0.5)
    w_losses = st.slider("Niederlagen", 0.0, 5.0, 1.5, 0.5)
    w_winrate = st.slider("Siegquote", 0.0, 5.0, 1.0, 0.5)

    st.download_button(
        "Excel-Vorlage herunterladen",
        data=template_excel(),
        file_name="muay_thai_matchmaker_vorlage.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

settings = MatchSettings(
    max_weight_diff=float(max_weight_diff),
    max_fight_diff=int(max_fight_diff),
    max_win_diff=int(max_win_diff),
    max_loss_diff=int(max_loss_diff),
    min_score=float(min_score),
    same_gender_only=bool(same_gender_only),
    weights=Weights(w_weight, w_fights, w_wins, w_losses, w_winrate),
)

uploaded_file = st.file_uploader("Excel-Datei hochladen (.xlsx)", type=["xlsx"])

st.info("Pflichtspalten: Name, Verein, Geschlecht, Gewicht, Kämpfe, Siege, Niederlagen")

if uploaded_file is not None:
    try:
        raw_df = pd.read_excel(uploaded_file)
        fighters, validation_errors = clean_fighters(raw_df)
        if validation_errors:
            st.error("Bitte korrigiere die Importdatei.")
            for err in validation_errors[:50]:
                st.write("- " + err)
            if len(validation_errors) > 50:
                st.write(f"… und {len(validation_errors) - 50} weitere Fehler.")
            st.stop()

        st.subheader("Importierte Kämpfer")
        st.dataframe(fighters[REQUIRED_COLUMNS], use_container_width=True, hide_index=True)

        matches_df, unmatched_df, candidates_df = build_matches(fighters, settings)

        col1, col2, col3 = st.columns(3)
        col1.metric("Kämpfer", len(fighters))
        col2.metric("Matches", len(matches_df))
        col3.metric("Nicht gematcht", len(unmatched_df))

        st.subheader("Optimale Paarungen")
        st.caption("Die Optimierung maximiert zuerst die Anzahl der Paarungen und danach den kombinierten Match-Score.")
        if matches_df.empty:
            st.warning("Keine gültigen Paarungen gefunden. Erhöhe die Toleranzen oder senke den Mindest-Score.")
        else:
            st.dataframe(matches_df, use_container_width=True, hide_index=True)

        with st.expander("Nicht gematchte Kämpfer"):
            if unmatched_df.empty:
                st.success("Alle Kämpfer wurden gematcht.")
            else:
                st.dataframe(unmatched_df, use_container_width=True, hide_index=True)

        with st.expander("Alle gültigen Kandidatenpaare"):
            if candidates_df.empty:
                st.write("Keine gültigen Kandidatenpaare gefunden.")
            else:
                st.dataframe(candidates_df.head(250), use_container_width=True, hide_index=True)

        export_bytes = export_to_excel(matches_df, unmatched_df, candidates_df)
        st.download_button(
            "Ergebnis als Excel herunterladen",
            data=export_bytes,
            file_name="muay_thai_matches.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as exc:
        st.error(f"Import oder Matching fehlgeschlagen: {exc}")
else:
    st.write("Lade eine Excel-Datei hoch oder nutze zuerst die Vorlage aus der Seitenleiste.")

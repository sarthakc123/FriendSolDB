import os
import re
import io
import base64
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

# -------------------------
# Config
# -------------------------
IMAGE_DIR = "molecule_images"   # folder to save PNGs

# -------------------------
# Load & clean
# -------------------------
df = pd.read_csv(
    "bigsoldb.csv",
    dtype={
        "SMILES_Solute": str,
        "Temperature_K": float,
        "Solvent": str,
        "SMILES_Solvent": str,
        "Solubility(mole_fraction)": float,
        "Solubility(mol/L)": float,
        "LogS(mol/L)": float,
        "Compound_Name": str,
    },
)

df_cleaned = df.dropna(subset=["LogS(mol/L)"])

# Focus on DMSO subset
df_dmso = df_cleaned.loc[df_cleaned["Solvent"] == "DMSO"].copy()

# -------------------------
# RDKit helpers
# -------------------------
def mol_to_data_url(mol, size=(500, 500)):
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

def sanitize_filename(name: str) -> str:
    """Safe-ish filename component."""
    if name is None or pd.isna(name):
        return ""
    name = name.strip()
    name = re.sub(r"[^\w\-\.]+", "_", name)
    return name[:80]

# -------------------------
# Build molecules & formulas (DMSO view)
# -------------------------
df_dmso["Structure"] = df_dmso["SMILES_Solute"].apply(Chem.MolFromSmiles)
df_dmso["Formula"] = df_dmso["Structure"].apply(lambda m: CalcMolFormula(m) if m else None)
df_dmso["img_data_url"] = df_dmso["Structure"].apply(mol_to_data_url)

# Assign permanent global index (for DMSO view)
df_dmso = df_dmso.reset_index(drop=True)
df_dmso["Idx"] = df_dmso.index + 1

# -------------------------
# Directly save all images (index-only filenames)
# -------------------------
os.makedirs(IMAGE_DIR, exist_ok=True)
saved = 0
# for _, row in df_dmso.iterrows():
#     mol = row["Structure"]
#     if mol is None:
#         continue
#     idx = int(row["Idx"])
#     path = os.path.join(IMAGE_DIR, f"mol_{idx:05d}.png")
#     Draw.MolToFile(mol, path, size=(500, 500))
#     saved += 1

# -------------------------
# Unique solvent count per solute (entire dataset)
# -------------------------
# Build a table with: SMILES_Solute | Compound_Name (first non-null) | Unique_Solvents | Solvent_List
def first_non_null(series):
    for x in series:
        if pd.notna(x) and str(x).strip() != "":
            return x
    return None

counts_df = (
    df_dmso.groupby("Compound_Name", dropna=False)
           .size()
           .reset_index(name="Count")
           .sort_values("Count", ascending=False)
           .reset_index(drop=True)
)

# -------------------------
# Display
# -------------------------

# -------------------------
# Sidebar: Temperature filter
# -------------------------
st.sidebar.header("Filters")

tmin = float(np.nanmin(df_dmso["Temperature_K"]))
tmax = float(np.nanmax(df_dmso["Temperature_K"]))
temp_range = st.sidebar.slider(
    "Temperature (K) range",
    min_value=float(np.floor(tmin)),
    max_value=float(np.ceil(tmax)),
    value=(float(np.floor(tmin)), float(np.ceil(tmax))),
    step=1.0,
)

lowK, highK = temp_range
# Filtered view for the MAIN DMSO TABLE only
mask = (df_dmso["Temperature_K"] >= lowK) & (df_dmso["Temperature_K"] <= highK)
df_view = df_dmso.loc[mask].copy().reset_index(drop=True)

# -------------------------
# Display table (FILTERED BY TEMPERATURE)
# -------------------------
st.title(f"Molecules in DMSO (Temperature {lowK:.0f}–{highK:.0f} K)")
st.dataframe(
    df_view[
        [
            "Idx",
            "SMILES_Solute",
            "Temperature_K",
            "Solvent",
            "SMILES_Solvent",
            "Solubility(mole_fraction)",
            "Solubility(mol/L)",
            "LogS(mol/L)",
            "Compound_Name",
            "Formula",
            "img_data_url",
        ]
    ],
    column_config={
        "img_data_url": st.column_config.ImageColumn("Structure", help="RDKit molecule"),
        "Idx": st.column_config.NumberColumn("Index"),
    },
    hide_index=True,
)

# -------------------------
# Per-solute counts (GLOBAL, UNFILTERED)
# -------------------------
counts_df = (
    df_dmso.groupby("SMILES_Solute", dropna=False)
           .size()
           .reset_index(name="Count")
           .sort_values("Count", ascending=False)
           .reset_index(drop=True)
)

st.subheader("Rows per Solute (Global DMSO counts)")
st.dataframe(
    counts_df,
    hide_index=True,
    column_config={
        "SMILES_Solute": st.column_config.TextColumn("Solute (SMILES)"),
        "Count": st.column_config.NumberColumn("Count"),
    },
)


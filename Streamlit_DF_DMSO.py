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
from rdkit.Chem import rdDepictor

# -------------------------
# Config
# -------------------------
IMAGE_DIR = "molecule_images"   # folder to save PNGs
THUMB_SIZE = (280, 280)   # smaller = faster
PAGE_SIZE_DEFAULT = 50    # rows per page in the table
# -------------------------
# Load & clean
# -------------------------
@st.cache_data(show_spinner=False)
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
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
    return df.dropna(subset=["LogS(mol/L)"])

@st.cache_data(show_spinner=False)
def smiles_to_mol_with_coords(smiles: str):
    """Parse once and compute 2D coords; cache by SMILES string."""
    if smiles is None or pd.isna(smiles):
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    rdDepictor.Compute2DCoords(m)
    return Chem.Mol(m)  # return a copy that is serializable

@st.cache_data(show_spinner=False)
def mol_png_data_url(smiles: str, size=(280, 280)) -> str | None:
    """Cached PNG (as data URL) per SMILES+size."""
    m = smiles_to_mol_with_coords(smiles)
    if m is None:
        return None
    img = Draw.MolToImage(m, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

@st.cache_data(show_spinner=False)
def compute_global_counts(df_dmso: pd.DataFrame) -> pd.DataFrame:
    return (
        df_dmso.groupby("SMILES_Solute", dropna=False)
               .size()
               .reset_index(name="Count")
               .sort_values("Count", ascending=False)
               .reset_index(drop=True)
    )

# -------------------------
# Load & subset
# -------------------------
df_cleaned = load_df("bigsoldb.csv")
df_dmso = df_cleaned.loc[df_cleaned["Solvent"] == "DMSO"].copy()

# Assign permanent global index once
df_dmso = df_dmso.reset_index(drop=True)
df_dmso["Idx"] = df_dmso.index + 1

# -------------------------
# Build *light* columns using cached functions
# (delay heavy work until needed; use SMILES -> cached image)
# -------------------------
# Formula is quick; keep it. Image uses cached data-url per unique SMILES.
df_dmso["Formula"] = df_dmso["SMILES_Solute"].apply(
    lambda s: CalcMolFormula(smiles_to_mol_with_coords(s)) if s else None
)
df_dmso["img_data_url"] = df_dmso["SMILES_Solute"].apply(
    lambda s: mol_png_data_url(s, size=THUMB_SIZE)
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

lowK = st.sidebar.number_input(
    "Min Temperature (K)",
    min_value=float(np.floor(tmin)),
    max_value=float(np.ceil(tmax)),
    value=float(np.floor(tmin)),
    step=1.0,
)

highK = st.sidebar.number_input(
    "Max Temperature (K)",
    min_value=float(np.floor(tmin)),
    max_value=float(np.ceil(tmax)),
    value=float(np.ceil(tmax)),
    step=1.0,
)

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
    df_dmso.groupby("Compound_Name", dropna=False)
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
        "Index": st.column_config.NumberColumn("Idx"),
        "Compound_Name": st.column_config.TextColumn("Compound_Name"),
        "Count": st.column_config.NumberColumn("Count"),
    },
)

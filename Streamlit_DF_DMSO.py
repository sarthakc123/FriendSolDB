import streamlit as st
import pandas as pd
import base64, io
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import numpy as np

df=pd.read_csv('bigsoldb.csv', dtype={'SMILES_Solute':str,'Temperature_K':float,'Solvent':str,'SMILES_Solvent':str,'Solubility(mole_fraction)':float,'Solubility(mol/L)':float,'LogS(mol/L)':float,'Compound_Name':str})

df_cleaned=df.dropna(subset=['LogS(mol/L)'])

num_cols = df_cleaned.select_dtypes(include=[np.number]).columns

stats = (
    df_cleaned[num_cols]
    .agg(['mean', 'std','median'])       # std uses sample std (ddof=1). Use .std(ddof=0) for population.
    .T
    .rename(columns={'mean': 'Mean', 'std': 'StdDev'})
    .sort_index()
)

print(stats.round(2))

# Filter
df_dmso = df_cleaned.loc[df_cleaned['Solvent'] == 'DMSO'].copy()

# Build RDKit molecules + formulas
df_dmso['Structure'] = df_dmso['SMILES_Solute'].apply(Chem.MolFromSmiles)
df_dmso['Formula'] = df_dmso['Structure'].apply(lambda m: CalcMolFormula(m) if m else None)

# Helper: mol → base64 image
def mol_to_data_url(mol, size=(500,500)):
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

df_dmso['img_data_url'] = df_dmso['Structure'].apply(mol_to_data_url)

# Display in a Streamlit table
st.title("Molecules in DMSO")
st.dataframe(
    df_dmso[['SMILES_Solute','Temperature_K','Solvent','SMILES_Solvent','Solubility(mole_fraction)','Solubility(mol/L)','LogS(mol/L)','Compound_Name','Formula','img_data_url']],
    column_config={
        "img_data_url": st.column_config.ImageColumn("Structure", help="RDKit molecule")
    },
    hide_index=True,
)
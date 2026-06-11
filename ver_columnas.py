import pandas as pd

snr = pd.read_csv(
    '01_SNR/fuentes/snr_actos/Actos_Jurídicos_Notariales_20260609.csv',
    encoding='latin-1', low_memory=False)

print("COLUMNAS DEL CSV:")
for c in snr.columns:
    print(f"  - {c}")

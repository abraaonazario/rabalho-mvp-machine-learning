import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

print("Baixando dataset completo...")
raw = fetch_covtype(as_frame=True)
X_full = raw.data.copy()
y_full = raw.target.copy()

print("Subamostrando 60.000 registros...")
X_sub, _, y_sub, _ = train_test_split(
    X_full, y_full,
    train_size=60000,
    random_state=42,
    stratify=y_full
)

# Renomear colunas
CONTINUOUS_FEATURES = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon",
    "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points",
]
WILDERNESS_FEATURES = [f"Wilderness_Area{i}" for i in range(1, 5)]
SOIL_FEATURES = [f"Soil_Type{i}" for i in range(1, 41)]

orig_cols = list(X_sub.columns)
new_cols = CONTINUOUS_FEATURES + WILDERNESS_FEATURES + SOIL_FEATURES
X_sub = X_sub.rename(columns=dict(zip(orig_cols, new_cols)))

X_sub["Cover_Type"] = y_sub.values

print("Salvando CSV...")
X_sub.to_csv("forest_cover_sample.csv", index=False)
print("Concluído: forest_cover_sample.csv")

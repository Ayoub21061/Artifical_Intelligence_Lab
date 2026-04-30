import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler


csv_path = Path(__file__).resolve().parent / "student_dataset" / "student_failure" / "train.csv"
df = pd.read_csv(csv_path)

print(df.head())
print(df.info())
print(df.describe())

# Distribution du score
plt.figure()
sns.histplot(df["score_examen"], bins=30)
plt.title("Distribution des scores")
plt.show()

# Corrélation
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Missing values
df["accès_internet"] = df["accès_internet"].fillna("unknown")
df["heures_etude"] = df["heures_etude"].fillna(df["heures_etude"].mean())

# Encodage des variables catégorielles - Encoding 
# Permet de convertir du texte en nombres. Dans notre dataset par exemple ça permet de faire : 
# Male = 1 et Female = 0
# drop_first=True permet d'éviter la multicolinéarité en supprimant une des catégories après l'encodage
# Permet d'éviter la redondance 
df = pd.get_dummies(df, drop_first=True)

# Scaling 
# Sans ça, les échelles sont très différentes 
# Avec le scaling, toute les variables deviennent comparables 
# Toute les variables ont le même poids et le modèle peut mieux apprendre. 
scaler = StandardScaler()
X = df.drop("score_examen", axis=1)
y = df["score_examen"]

X_scaled = scaler.fit_transform(X)
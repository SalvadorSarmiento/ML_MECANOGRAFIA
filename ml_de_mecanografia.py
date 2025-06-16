import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import os

# === RUTAS ===
ruta_csv = "C:\\Users\\eduar\\OneDrive\\Documentos\\ML_mecanografia\\datos_procesados.csv"
ruta_metricas = "C:\\Users\\eduar\\OneDrive\\Documentos\\ML_mecanografia\\metricas_comparativas.csv"
ruta_dir_roc = "C:\\Users\\eduar\\OneDrive\\Documentos\\ML_mecanografia\\curvas_roc_individuales"

# Crear carpeta para curvas si no existe
os.makedirs(ruta_dir_roc, exist_ok=True)

# === CARGA Y PROCESAMIENTO DE DATOS ===
df = pd.read_csv(ruta_csv)

# Filtrar entradas inválidas
def texto_valido(texto):
    import re
    if not isinstance(texto, str) or len(texto.strip()) < 5:
        return False
    return not re.fullmatch(r"[xsl*]+", texto.strip().lower())

df = df[df["textoEscrito"].apply(texto_valido)]

# Calcular errores y etiqueta
def contar_errores(fila):
    escrito = fila["textoEscrito"].split()
    mostrado = fila["textoMostrado"].split()
    return sum(1 for a, b in zip(escrito, mostrado) if a != b)

df["errores"] = df.apply(contar_errores, axis=1)
df["completo"] = (df["textoEscrito"].str.strip() == df["textoMostrado"].str.strip()).astype(int)

# === VARIABLES ===
X = df[["nivel", "tiempo", "errores"]]
y = df["completo"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# === MODELOS ===
modelos = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "RandomForest": RandomForestClassifier(),
    "LightGBM": LGBMClassifier(),
    "SVM": SVC(probability=True),
    "MLP": MLPClassifier(max_iter=500),
    "Stacking": StackingClassifier(
        estimators=[
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
            ('rf', RandomForestClassifier()),
            ('svm', SVC(probability=True))
        ],
        final_estimator=LogisticRegression()
    )
}

# === MÉTRICAS Y ROC AUC INDIVIDUAL ===
metricas = []

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

        # Guardar curva ROC individual, por modelo.
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color='blue', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"Curva ROC - {nombre}")
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    # Guardar con nombre limpio
    nombre_archivo = f"roc_{nombre.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(ruta_dir_roc, nombre_archivo))
    plt.close()

    # Agregar métricas
    metricas.append({
        "Modelo": nombre,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "AUC ROC": auc
    })

# === GUARDAR MÉTRICAS EN CSV Y GRAFICAR ===
df_metricas = pd.DataFrame(metricas)
df_metricas.to_csv(ruta_metricas, index=False)

# === GRAFICAR MÉTRICAS COMPARATIVAS ===
plt.figure(figsize=(12, 6))
df_melt = df_metricas.melt(id_vars="Modelo", var_name="Métrica", value_name="Valor")

sns.barplot(data=df_melt, x="Modelo", y="Valor", hue="Métrica")
plt.title("Comparación de Métricas de Modelos")
plt.ylim(0, 1.05)
plt.xticks(rotation=30)
plt.tight_layout()
plt.grid(axis="y")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()

# === CURVAS ROC COMPARATIVAS EN UNA SOLA GRÁFICA ===
plt.figure(figsize=(10, 8))

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{nombre} (AUC={auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("Curvas ROC Comparativas")
plt.xlabel("Falsos Positivos")
plt.ylabel("Verdaderos Positivos")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("C:\\Users\\eduar\\OneDrive\\Documentos\\ML_mecanografia\\roc_comparativo.png")
plt.show()
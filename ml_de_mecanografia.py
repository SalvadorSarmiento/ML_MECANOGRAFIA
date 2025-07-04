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
from scipy.stats import percentileofscore

# === RUTAS ===
ruta_csv = "C:\\Users\\eduar\\OneDrive\\Documentos\\ML_mecanografia\\datos_procesados.csv"
ruta_metricas = "C:\\Users\\eduar\\OneDrive\\Documentos\\ML_mecanografia\\metricas_comparativas.csv"
ruta_dir_roc = "C:\\Users\\eduar\\OneDrive\\Documentos\\ML_mecanografia\\curvas_roc_individuales"

# Crear carpeta para curvas roc
os.makedirs(ruta_dir_roc, exist_ok=True)

# === CARGA, PROCESAMIENTO DE DATOS ===
df = pd.read_csv(ruta_csv)

# Filtro de entradas invalidas
def texto_valido(texto):
    import re
    if not isinstance(texto, str) or len(texto.strip()) < 5:
        return False
    return not re.fullmatch(r"[xsl*]+", texto.strip().lower())

df = df[df["textoEscrito"].apply(texto_valido)]

# === Conteo de usuarios por CORREO ===
total_usuarios = df["correo"].nunique()
print(f"Total de usuarios únicos en el dataset: {total_usuarios}")

# === Listado de usuarios por correo ===
usuarios_unicos = df["correo"].unique()
print("Usuarios únicos encontrados:")
for correo in usuarios_unicos:
    print(correo)

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

# === MÉTRICAS y ROC AUC Por Modelo ===
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

    # Guardar curva ROC individual
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color='blue', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"Curva ROC - {nombre}")
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    nombre_archivo = f"roc_{nombre.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(ruta_dir_roc, nombre_archivo))
    plt.close()

    # Adición de métricas
    metricas.append({
        "Modelo": nombre,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "AUC ROC": auc
    })

    # Mostrar métricas en consola
    print(f"\nModelo: {nombre}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  AUC ROC  : {auc:.4f}")


# === Guardado de metricas en CSV y grafico ===
df_metricas = pd.DataFrame(metricas)
df_metricas.to_csv(ruta_metricas, index=False)

# === Grafico de barras por métricas ===
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

# === Curvas ROC comparadas ===
plt.figure(figsize=(10, 8))

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{nombre} (AUC={auc:.2f})")

# === Normalizar errores y tiempo en Z-score ===
df["errores_z"] = (df["errores"] - df["errores"].mean()) / df["errores"].std()
df["tiempo_z"] = (df["tiempo"] - df["tiempo"].mean()) / df["tiempo"].std()

# === Nivel de estrés captado: promedio de Z-score por correo ===
df_estres = (
    df.groupby("correo", sort=False)  # sort=False mantiene el orden en pandas 1.1+
    .agg(
        errores_z_mean=("errores_z", "mean"),
        tiempo_z_mean=("tiempo_z", "mean")
    )
    .copy()
)

# Nivel de estrés final: media de errores_z y tiempo_z
df_estres["nivel_estres"] = df_estres[["errores_z_mean", "tiempo_z_mean"]].mean(axis=1)

# Percentil del nivel de estrés
todos_valores = df_estres["nivel_estres"].values
df_estres["percentil_estres"] = df_estres["nivel_estres"].apply(
    lambda x: percentileofscore(todos_valores, x)
)

# Restaurar orden de aparición original
orden_original = df["correo"].drop_duplicates().tolist()
df_estres = df_estres.reindex(orden_original).reset_index()

# Asignar categoría de estrés según percentil
def categorizar(percentil):
    if percentil <= 33:
        return "Bajo"
    elif percentil <= 66:
        return "Medio"
    else:
        return "Alto"

df_estres["categoria_estres"] = df_estres["percentil_estres"].apply(categorizar)

# Enumeración
df_estres.insert(0, "Nro", range(1, len(df_estres) + 1))

# Selección de columnas para CSV
df_reporte = df_estres[["Nro", "correo", "nivel_estres", "percentil_estres", "categoria_estres"]]

# === Guarda archivo CSV ===
ruta_reporte_estres = "C:\\Users\\eduar\\OneDrive\\Documentos\\ML_mecanografia\\reporte_estres_estudiantes.csv"
df_reporte.to_csv(ruta_reporte_estres, index=False)

print(f"\nReporte de estrés individual guardado en:\n{ruta_reporte_estres}")

# === Generar gráfico de barras ===

# Preparación de datos
correos = df_estres["correo"]
percentiles = df_estres["percentil_estres"]

# Colores de percentil
colores = []
for p in percentiles:
    if p <= 33:
        colores.append("green")  # Bajo
    elif p <= 66:
        colores.append("orange") # Medio
    else:
        colores.append("red")    # Alto

# Crear figura
fig, ax = plt.subplots(figsize=(10, len(correos)*0.4))

# Barras horizontales
bars = ax.barh(correos, percentiles, color=colores)

# Etiquetas con número sin decimales
for bar, p in zip(bars, percentiles):
    ax.text(p + 1, bar.get_y() + bar.get_height()/2,
            f"{int(round(p, 0))}%", va="center", fontsize=9)

# Límites y etiquetas
ax.set_xlim(0, 100)
ax.set_xlabel("Percentil de Estrés")
ax.set_title("Nivel de Estrés por Estudiante")

# Leyenda manual
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="green", label="Bajo (≤33%)"),
    Patch(facecolor="orange", label="Medio (34–66%)"),
    Patch(facecolor="red", label="Alto (≥67%)")
]
ax.legend(handles=legend_elements, loc="lower right")

# Ajustar layout
plt.tight_layout()

# Guardar imagen
ruta_grafico = "C:\\Users\\eduar\\OneDrive\\Documentos\\ML_mecanografia\\grafico_estres_estudiantes.png"
plt.savefig(ruta_grafico, dpi=300)
plt.close()

print(f"\nGráfico de estrés guardado en:\n{ruta_grafico}")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("Curvas ROC Comparativas")
plt.xlabel("Falsos Positivos")
plt.ylabel("Verdaderos Positivos")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("C:\\Users\\eduar\\OneDrive\\Documentos\\ML_mecanografia\\roc_comparativo.png")
plt.show()

# === Guardado de lista de usuarios por correo en CSV ===
df_usuarios = pd.DataFrame({"correo": usuarios_unicos})
df_usuarios.to_csv("C:\\Users\\eduar\\OneDrive\\Documentos\\ML_mecanografia\\usuarios_unicos.csv", index=False)
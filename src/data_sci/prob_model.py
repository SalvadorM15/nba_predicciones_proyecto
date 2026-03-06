import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, classification_report

def load_and_split_data(dataset_path):
    """
    Carga el dataset de entrenamiento, separa features (X) de target (y)
    y divide en conjuntos de entrenamiento y prueba.
    """
    df = pd.read_csv(dataset_path)
    
    # Separamos y (Target)
    y = df['Target_Local_Win'].values
    
    # Separamos X (Features). Descartamos columnas informativas categóricas.
    cols_to_drop = ['Game_ID', 'GAME_DATE', 'Local_team_id', 'Visitor_team_id', 'Target_Local_Win']
    X = df.drop(columns=cols_to_drop)
    
    # Dividimos 80% entrenamiento, 20% prueba. Usamos shuffle=False si queremos mantener orden cronológico
    # o suffle=True para mezclar. Para empezar, mezclamos asegurando un estado aleatorio.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, X.columns

def train_logistic_model(X_train, y_train):
    """
    Crea un pipeline que estandariza los datos y luego entrena una regresión logística.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test, report_path=None, verbose=True):
    """
    Evalúa el modelo usando el conjunto de prueba e imprime métricas clave.
    Si se especifica un report_path, guardará un archivo de texto con la documentación.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)
    
    report_text = "--- Resultados de la Evaluación del Modelo ---\n"
    report_text += f"Accuracy (Precisión Global): {acc:.4f}\n"
    report_text += f"Brier Score (Puntuación de Calibración): {brier:.4f}\n\n"
    report_text += "Matriz de Confusión:\n"
    report_text += str(confusion_matrix(y_test, y_pred)) + "\n\n"
    report_text += "Reporte de Clasificación:\n"
    report_text += classification_report(y_test, y_pred) + "\n"
    
    if verbose:
        print(report_text)
    
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        if verbose:
            print(f"[+] Informe de evaluación guardado en: {report_path}")
    
    return acc, brier

def save_model(model, filepath, verbose=True):
    """
    Guarda el modelo entrenado en disco.
    """
    joblib.dump(model, filepath)
    if verbose:
        print(f"[+] Modelo guardado exitosamente en: {filepath}")

def load_model(filepath):
    """
    Carga un modelo desde el disco.
    """
    return joblib.load(filepath)

def predict_game(model, match_features_df):
    """
    Dada una fila de estadísticas generada, calcula la probabilidad de victoria local.
    """
    prob_win = model.predict_proba(match_features_df)[0][1]
    return prob_win
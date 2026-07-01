# Databricks notebook source
# MAGIC %md
# MAGIC # Demo autocontenida: estrategias de rollout con Mosaic AI Model Serving
# MAGIC
# MAGIC Este notebook esta disenado para una charla de MLOps en Databricks. Replica las ideas centrales de una demo de **Model Rollout Strategies with Mosaic AI Model Serving**, pero sin depender de notebooks de setup, objetos de Databricks Academy, modelos registrados existentes, endpoints existentes, tablas externas ni archivos del curso.
# MAGIC
# MAGIC En esta demo vamos a:
# MAGIC
# MAGIC - Crear un dataset sintetico de churn con Spark.
# MAGIC - Entrenar dos versiones de modelo: `champion` y `challenger`.
# MAGIC - Registrar metricas y modelos en MLflow cuando este disponible.
# MAGIC - Construir configuraciones dry-run de Mosaic AI Model Serving para 100%, canary, A/B, blue-green y rollback.
# MAGIC - Simular trafico de inferencia para observar ruteo, metricas y guardrails.
# MAGIC - Mantener una seccion opcional para despliegue real, desactivada por defecto para que `Run all` no cree recursos ni consuma serving compute.

# COMMAND ----------

# MAGIC %pip install -q -U scikit-learn pandas mlflow databricks-sdk

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuracion de la demo
# MAGIC
# MAGIC El flujo principal es autocontenido y seguro para Serverless: genera datos, entrena modelos, simula ruteo y produce payloads de configuracion. No crea endpoints reales a menos que cambies explicitamente `ACTIVAR_DESPLIEGUE_REAL` a `True`.

# COMMAND ----------

import hashlib
import json
import os
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

SEED = 42
NUM_FILAS = 12000
LABEL_COL = "will_churn"
FEATURE_COLS = [
    "account_age_days",
    "monthly_spend",
    "support_tickets_30d",
    "usage_minutes_30d",
    "failed_payments_90d",
    "contract_value",
    "discount_pct",
    "team_seats",
    "feature_adoption_score",
    "days_since_last_login",
]


def mostrar(objeto, n=10):
    """Muestra Spark DataFrames, Pandas DataFrames u objetos simples."""
    try:
        if hasattr(objeto, "limit"):
            display(objeto.limit(n))
        elif isinstance(objeto, pd.DataFrame):
            display(objeto.head(n))
        else:
            display(objeto)
    except Exception:
        if hasattr(objeto, "show"):
            objeto.show(n, truncate=False)
        else:
            print(objeto)


def obtener_directorio_notebook(default="/Shared"):
    """Obtiene el directorio actual del notebook sin depender del curso."""
    try:
        notebook_path = (
            dbutils.notebook.entry_point.getDbutils()
            .notebook()
            .getContext()
            .notebookPath()
            .get()
        )
        return os.path.dirname(notebook_path)
    except Exception:
        return default


MLFLOW_ACTIVO = False
EXPERIMENT_ID = None
EXPERIMENT_NAME = (
    f"{obtener_directorio_notebook()}/2.3b - Demo Serverless Rollout Strategies"
)

try:
    import mlflow

    mlflow.set_experiment(EXPERIMENT_NAME)
    experimento = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    EXPERIMENT_ID = experimento.experiment_id if experimento else None
    MLFLOW_ACTIVO = True

    if mlflow.active_run() is not None:
        mlflow.end_run()

    print(f"Experimento MLflow: {EXPERIMENT_NAME}")
except Exception as exc:
    print(f"MLflow no esta disponible para tracking en esta sesion: {exc}")


def iniciar_run(nombre):
    """Abre un run de MLflow cuando esta disponible; si no, usa un contexto vacio."""
    if not MLFLOW_ACTIVO:
        return nullcontext()

    kwargs = {"run_name": nombre}
    if EXPERIMENT_ID:
        kwargs["experiment_id"] = EXPERIMENT_ID
    return mlflow.start_run(**kwargs)


def registrar_metricas(metricas):
    if not MLFLOW_ACTIVO:
        return

    for clave, valor in metricas.items():
        try:
            mlflow.log_metric(clave, float(valor))
        except Exception as exc:
            print(f"No se pudo registrar la metrica {clave}: {exc}")


def registrar_parametros(parametros):
    if not MLFLOW_ACTIVO:
        return

    for clave, valor in parametros.items():
        try:
            mlflow.log_param(clave, valor)
        except Exception as exc:
            print(f"No se pudo registrar el parametro {clave}: {exc}")


def registrar_texto(texto, ruta):
    if not MLFLOW_ACTIVO:
        return

    try:
        mlflow.log_text(texto, ruta)
    except Exception as exc:
        print(f"No se pudo registrar {ruta}: {exc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Crear datos sinteticos de churn con Spark
# MAGIC
# MAGIC El dataset representa clientes SaaS. La etiqueta `will_churn` se genera con una senal controlada: mas tickets, pagos fallidos y dias sin login aumentan el riesgo; mas adopcion del producto, uso y antiguedad reducen el riesgo.

# COMMAND ----------

clientes_df = (
    spark.range(NUM_FILAS)
    .withColumn("account_age_days", F.round(F.lit(30) + F.rand(SEED + 1) * F.lit(1500), 0))
    .withColumn("monthly_spend", F.round(F.lit(20) + F.rand(SEED + 2) * F.lit(780), 2))
    .withColumn("support_tickets_30d", F.floor(F.rand(SEED + 3) * F.lit(8)).cast("double"))
    .withColumn("usage_minutes_30d", F.round(F.lit(20) + F.rand(SEED + 4) * F.lit(2500), 2))
    .withColumn("failed_payments_90d", F.floor(F.rand(SEED + 5) * F.lit(4)).cast("double"))
    .withColumn("contract_value", F.round(F.lit(500) + F.rand(SEED + 6) * F.lit(49500), 2))
    .withColumn("discount_pct", F.round(F.rand(SEED + 7) * F.lit(0.45), 3))
    .withColumn("team_seats", F.floor(F.lit(1) + F.rand(SEED + 8) * F.lit(250)).cast("double"))
    .withColumn("feature_adoption_score", F.round(F.rand(SEED + 9), 4))
    .withColumn("days_since_last_login", F.round(F.rand(SEED + 10) * F.lit(90), 0))
)

logit = (
    F.lit(-1.4)
    + F.col("support_tickets_30d") * F.lit(0.38)
    + F.col("failed_payments_90d") * F.lit(0.65)
    + F.col("days_since_last_login") * F.lit(0.035)
    + F.col("discount_pct") * F.lit(1.10)
    - F.col("feature_adoption_score") * F.lit(2.20)
    - F.log(F.col("usage_minutes_30d") + F.lit(1.0)) * F.lit(0.22)
    - F.log(F.col("account_age_days") + F.lit(1.0)) * F.lit(0.16)
    + F.randn(SEED + 11) * F.lit(0.35)
)

clientes_df = (
    clientes_df.withColumn("churn_probability", F.lit(1.0) / (F.lit(1.0) + F.exp(-logit)))
    .withColumn(LABEL_COL, (F.rand(SEED + 12) < F.col("churn_probability")).cast("int"))
    .drop("id")
)

print(f"Filas generadas: {clientes_df.count():,}")
mostrar(clientes_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Preparar datos para entrenamiento local
# MAGIC
# MAGIC Para mantener la demo rapida y portable, entrenamos modelos scikit-learn en memoria local. Spark se encarga de crear y preparar los datos; MLflow se encarga de la trazabilidad.

# COMMAND ----------

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

clientes_pdf = clientes_df.select(FEATURE_COLS + [LABEL_COL]).toPandas()

X = clientes_pdf[FEATURE_COLS]
y = clientes_pdf[LABEL_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=SEED,
    stratify=y,
)

print(f"Entrenamiento: {X_train.shape}")
print(f"Prueba:        {X_test.shape}")
print(f"Tasa churn train: {y_train.mean():.3f}")
print(f"Tasa churn test:  {y_test.mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Entrenar champion y challenger
# MAGIC
# MAGIC Simularemos dos versiones de un modelo:
# MAGIC
# MAGIC - `champion`: version estable, simple y explicable.
# MAGIC - `challenger`: version candidata, mas flexible y potencialmente mas precisa.
# MAGIC
# MAGIC En una implementacion real, estas versiones vivirian en Unity Catalog como versiones registradas de un mismo modelo. En esta demo las representamos con metadata local y runs de MLflow.

# COMMAND ----------

def evaluar_clasificador(nombre_modelo, modelo, X_eval, y_eval):
    probas = modelo.predict_proba(X_eval)[:, 1]
    predicciones = (probas >= 0.5).astype(int)
    return {
        "modelo": nombre_modelo,
        "auc_roc": float(roc_auc_score(y_eval, probas)),
        "average_precision": float(average_precision_score(y_eval, probas)),
        "accuracy": float(accuracy_score(y_eval, predicciones)),
        "f1": float(f1_score(y_eval, predicciones)),
        "log_loss": float(log_loss(y_eval, probas)),
    }


def registrar_modelo_sklearn(nombre_run, nombre_modelo, modelo, metricas, parametros):
    run_id = None

    with iniciar_run(nombre_run) as run:
        if run is not None:
            run_id = run.info.run_id

        registrar_parametros(parametros)
        registrar_metricas({k: v for k, v in metricas.items() if k != "modelo"})

        if MLFLOW_ACTIVO:
            try:
                from mlflow.models.signature import infer_signature

                firma = infer_signature(X_train.head(5), modelo.predict_proba(X_train.head(5)))
                mlflow.sklearn.log_model(
                    sk_model=modelo,
                    artifact_path="model",
                    signature=firma,
                    input_example=X_train.head(5),
                )
            except Exception as exc:
                print(f"No se pudo registrar el modelo {nombre_modelo}: {exc}")

    return run_id


champion_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                random_state=SEED,
            ),
        ),
    ]
)

challenger_model = HistGradientBoostingClassifier(
    max_iter=180,
    learning_rate=0.06,
    max_leaf_nodes=24,
    l2_regularization=0.05,
    random_state=SEED,
)

inicio = time.time()
champion_model.fit(X_train, y_train)
champion_training_duration = time.time() - inicio

inicio = time.time()
challenger_model.fit(X_train, y_train)
challenger_training_duration = time.time() - inicio

champion_metrics = evaluar_clasificador("champion", champion_model, X_test, y_test)
challenger_metrics = evaluar_clasificador("challenger", challenger_model, X_test, y_test)

champion_run_id = registrar_modelo_sklearn(
    "train_champion_v1",
    "champion",
    champion_model,
    {**champion_metrics, "training_duration_s": champion_training_duration},
    {
        "modelo_logico": "customer_churn_rollout_demo",
        "version_logica": "1",
        "rol": "champion",
        "estimator": "LogisticRegression",
    },
)

challenger_run_id = registrar_modelo_sklearn(
    "train_challenger_v2",
    "challenger",
    challenger_model,
    {**challenger_metrics, "training_duration_s": challenger_training_duration},
    {
        "modelo_logico": "customer_churn_rollout_demo",
        "version_logica": "2",
        "rol": "challenger",
        "estimator": "HistGradientBoostingClassifier",
    },
)

model_registry_simulado = pd.DataFrame(
    [
        {
            "modelo": "main.default.customer_churn_rollout_demo",
            "version": "1",
            "alias": "champion",
            "run_id": champion_run_id or "sin_mlflow",
            "auc_roc": champion_metrics["auc_roc"],
            "log_loss": champion_metrics["log_loss"],
        },
        {
            "modelo": "main.default.customer_churn_rollout_demo",
            "version": "2",
            "alias": "challenger",
            "run_id": challenger_run_id or "sin_mlflow",
            "auc_roc": challenger_metrics["auc_roc"],
            "log_loss": challenger_metrics["log_loss"],
        },
    ]
)

mostrar(model_registry_simulado)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Configuraciones dry-run para Mosaic AI Model Serving
# MAGIC
# MAGIC Un endpoint de serving puede enviar trafico a una o varias versiones. Aqui construimos configuraciones como diccionarios para poder explicarlas, versionarlas y revisarlas antes de tocar infraestructura real.
# MAGIC
# MAGIC Las rutas usan nombres logicos `churn_champion_v1` y `churn_challenger_v2`. En una demo real, el `model_name` debe apuntar a un modelo registrado al que tengas permisos de lectura y serving.

# COMMAND ----------

REGISTERED_MODEL_NAME = "mlops_dbx_talk_dev.ezapata.customer_churn_rollout_demo"
ENDPOINT_NAME = "demo-rollout-customer-churn"

SERVED_MODEL_CHAMPION = "churn_champion_v1"
SERVED_MODEL_CHALLENGER = "churn_challenger_v2"


def construir_config_serving(champion_pct, challenger_pct):
    rutas = []
    if champion_pct > 0:
        rutas.append(
            {
                "served_model_name": SERVED_MODEL_CHAMPION,
                "traffic_percentage": int(champion_pct),
            }
        )
    if challenger_pct > 0:
        rutas.append(
            {
                "served_model_name": SERVED_MODEL_CHALLENGER,
                "traffic_percentage": int(challenger_pct),
            }
        )

    return {
        "served_models": [
            {
                "name": SERVED_MODEL_CHAMPION,
                "model_name": REGISTERED_MODEL_NAME,
                "model_version": "1",
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
            },
            {
                "name": SERVED_MODEL_CHALLENGER,
                "model_name": REGISTERED_MODEL_NAME,
                "model_version": "2",
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
            },
        ],
        "traffic_config": {"routes": rutas},
    }


serving_configs = {
    "champion_100": construir_config_serving(100, 0),
    "canary_5": construir_config_serving(95, 5),
    "canary_25": construir_config_serving(75, 25),
    "ab_50_50": construir_config_serving(50, 50),
    "blue_green_challenger_100": construir_config_serving(0, 100),
    "rollback_champion_100": construir_config_serving(100, 0),
}

print(json.dumps(serving_configs["canary_5"], indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Simular trafico del endpoint
# MAGIC
# MAGIC En vez de crear un endpoint real, simulamos el comportamiento de ruteo. El hash de `request_id` decide si una solicitud va al champion o al challenger segun los porcentajes de trafico.
# MAGIC
# MAGIC Esto permite explicar canary, A/B testing y rollback con resultados observables, sin depender de permisos ni de serving compute.

# COMMAND ----------

MODELOS_SERVIDOS = {
    SERVED_MODEL_CHAMPION: {
        "rol": "champion",
        "modelo": champion_model,
        "latencia_media_ms": 22,
        "latencia_sd_ms": 5,
    },
    SERVED_MODEL_CHALLENGER: {
        "rol": "challenger",
        "modelo": challenger_model,
        "latencia_media_ms": 35,
        "latencia_sd_ms": 8,
    },
}


def bucket_estable(request_id, estrategia):
    llave = f"{estrategia}:{request_id}".encode("utf-8")
    return int(hashlib.sha256(llave).hexdigest(), 16) % 100


def elegir_modelo(request_id, estrategia, rutas):
    bucket = bucket_estable(request_id, estrategia)
    acumulado = 0

    for ruta in rutas:
        acumulado += int(ruta["traffic_percentage"])
        if bucket < acumulado:
            return ruta["served_model_name"]

    return rutas[-1]["served_model_name"]


def simular_rollout(nombre_estrategia, config, X_requests, y_requests):
    rutas = config["traffic_config"]["routes"]
    resultados = []
    rng = np.random.default_rng(SEED + len(nombre_estrategia))

    for posicion, (_, features) in enumerate(X_requests.iterrows()):
        request_id = f"req-{posicion:06d}"
        served_model_name = elegir_modelo(request_id, nombre_estrategia, rutas)
        metadata_modelo = MODELOS_SERVIDOS[served_model_name]
        modelo = metadata_modelo["modelo"]

        features_df = pd.DataFrame([features], columns=FEATURE_COLS)
        probabilidad = float(modelo.predict_proba(features_df)[0, 1])
        prediccion = int(probabilidad >= 0.5)
        latencia = max(
            1.0,
            rng.normal(
                loc=metadata_modelo["latencia_media_ms"],
                scale=metadata_modelo["latencia_sd_ms"],
            ),
        )

        resultados.append(
            {
                "request_id": request_id,
                "estrategia": nombre_estrategia,
                "served_model_name": served_model_name,
                "rol": metadata_modelo["rol"],
                "y_true": int(y_requests.iloc[posicion]),
                "prediccion": prediccion,
                "probabilidad_churn": probabilidad,
                "latencia_ms": float(latencia),
            }
        )

    return pd.DataFrame(resultados)


def resumir_rollout(resultados_pdf):
    resumen = []

    for (estrategia, rol), grupo in resultados_pdf.groupby(["estrategia", "rol"]):
        resumen.append(
            {
                "estrategia": estrategia,
                "rol": rol,
                "requests": int(len(grupo)),
                "traffic_observado_pct": float(len(grupo) / len(resultados_pdf) * 100),
                "auc_roc": float(roc_auc_score(grupo["y_true"], grupo["probabilidad_churn"]))
                if grupo["y_true"].nunique() > 1
                else None,
                "accuracy": float(accuracy_score(grupo["y_true"], grupo["prediccion"])),
                "f1": float(f1_score(grupo["y_true"], grupo["prediccion"], zero_division=0)),
                "latencia_p50_ms": float(grupo["latencia_ms"].quantile(0.50)),
                "latencia_p95_ms": float(grupo["latencia_ms"].quantile(0.95)),
                "tasa_predicha_churn": float(grupo["prediccion"].mean()),
            }
        )

    return pd.DataFrame(resumen)


X_requests = X_test.reset_index(drop=True).head(1200)
y_requests = y_test.reset_index(drop=True).head(1200)

resultados_rollout = {}
resumenes = []

for estrategia, config in serving_configs.items():
    resultados = simular_rollout(estrategia, config, X_requests, y_requests)
    resultados_rollout[estrategia] = resultados
    resumenes.append(resumir_rollout(resultados))

rollout_summary_pdf = pd.concat(resumenes, ignore_index=True)
mostrar(rollout_summary_pdf.sort_values(["estrategia", "rol"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Shadow deployment
# MAGIC
# MAGIC En un despliegue shadow, el usuario recibe respuesta del champion, pero el challenger tambien se evalua en paralelo para comparar comportamiento. Esto normalmente se implementa fuera del endpoint o en una capa de orquestacion.

# COMMAND ----------

def simular_shadow(X_requests, y_requests):
    champion_proba = champion_model.predict_proba(X_requests)[:, 1]
    challenger_proba = challenger_model.predict_proba(X_requests)[:, 1]

    champion_pred = (champion_proba >= 0.5).astype(int)
    challenger_pred = (challenger_proba >= 0.5).astype(int)

    return pd.DataFrame(
        {
            "request_id": [f"req-{idx:06d}" for idx in range(len(X_requests))],
            "y_true": y_requests.to_numpy(),
            "champion_probability": champion_proba,
            "challenger_probability": challenger_proba,
            "champion_prediction": champion_pred,
            "challenger_prediction": challenger_pred,
            "prediction_disagrees": champion_pred != challenger_pred,
            "abs_probability_delta": np.abs(champion_proba - challenger_proba),
        }
    )


shadow_pdf = simular_shadow(X_requests, y_requests)

shadow_metrics = {
    "champion_auc": float(roc_auc_score(shadow_pdf["y_true"], shadow_pdf["champion_probability"])),
    "challenger_auc": float(roc_auc_score(shadow_pdf["y_true"], shadow_pdf["challenger_probability"])),
    "prediction_disagreement_rate": float(shadow_pdf["prediction_disagrees"].mean()),
    "avg_probability_delta": float(shadow_pdf["abs_probability_delta"].mean()),
    "p95_probability_delta": float(shadow_pdf["abs_probability_delta"].quantile(0.95)),
}

with iniciar_run("shadow_validation"):
    registrar_metricas(shadow_metrics)
    registrar_texto(shadow_pdf.head(200).to_json(orient="records", indent=2), "shadow_sample.json")

mostrar(pd.DataFrame([shadow_metrics]))
mostrar(shadow_pdf.sort_values("abs_probability_delta", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Guardrails de promocion
# MAGIC
# MAGIC Los guardrails convierten metricas en una decision operacional. Aqui usamos reglas simples para la demo:
# MAGIC
# MAGIC - El AUC del challenger no debe caer mas de 0.01 contra el champion.
# MAGIC - La latencia p95 del challenger debe estar por debajo de 60 ms.
# MAGIC - La tasa de desacuerdo shadow debe estar por debajo de 25%.

# COMMAND ----------

champion_auc = champion_metrics["auc_roc"]
challenger_auc = challenger_metrics["auc_roc"]

canary_25_summary = rollout_summary_pdf[
    (rollout_summary_pdf["estrategia"] == "canary_25")
    & (rollout_summary_pdf["rol"] == "challenger")
].iloc[0]

guardrails = pd.DataFrame(
    [
        {
            "guardrail": "AUC challenger >= champion - 0.01",
            "valor": challenger_auc,
            "umbral": champion_auc - 0.01,
            "pasa": challenger_auc >= champion_auc - 0.01,
        },
        {
            "guardrail": "Latencia p95 challenger <= 60 ms",
            "valor": float(canary_25_summary["latencia_p95_ms"]),
            "umbral": 60.0,
            "pasa": float(canary_25_summary["latencia_p95_ms"]) <= 60.0,
        },
        {
            "guardrail": "Desacuerdo shadow <= 25%",
            "valor": shadow_metrics["prediction_disagreement_rate"],
            "umbral": 0.25,
            "pasa": shadow_metrics["prediction_disagreement_rate"] <= 0.25,
        },
    ]
)

decision_promocion = "promover_challenger" if bool(guardrails["pasa"].all()) else "mantener_champion"

with iniciar_run("rollout_guardrails"):
    registrar_parametros({"decision_promocion": decision_promocion})
    for indice, fila in enumerate(guardrails.itertuples(index=False), start=1):
        registrar_metricas({f"guardrail_{indice}_valor": fila.valor})
    registrar_texto(guardrails.to_json(orient="records", indent=2), "guardrails.json")

print(f"Decision sugerida: {decision_promocion}")
mostrar(guardrails)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Plan de rollout generado
# MAGIC
# MAGIC Esta tabla resume las fases que podrias presentar en la charla. En una ejecucion real, cada fila corresponde a una actualizacion del endpoint con la configuracion indicada.

# COMMAND ----------

rollout_plan = pd.DataFrame(
    [
        {
            "fase": 1,
            "nombre": "Baseline productivo",
            "config": "champion_100",
            "champion_pct": 100,
            "challenger_pct": 0,
            "criterio_siguiente": "Modelo estable y monitoreo base listo",
        },
        {
            "fase": 2,
            "nombre": "Shadow validation",
            "config": "champion_100",
            "champion_pct": 100,
            "challenger_pct": 0,
            "criterio_siguiente": "Bajo desacuerdo y metricas offline aceptables",
        },
        {
            "fase": 3,
            "nombre": "Canary pequeno",
            "config": "canary_5",
            "champion_pct": 95,
            "challenger_pct": 5,
            "criterio_siguiente": "Sin regresion de calidad ni latencia",
        },
        {
            "fase": 4,
            "nombre": "Canary ampliado",
            "config": "canary_25",
            "champion_pct": 75,
            "challenger_pct": 25,
            "criterio_siguiente": "Guardrails pasan",
        },
        {
            "fase": 5,
            "nombre": "A/B controlado",
            "config": "ab_50_50",
            "champion_pct": 50,
            "challenger_pct": 50,
            "criterio_siguiente": "Impacto negocio positivo",
        },
        {
            "fase": 6,
            "nombre": "Blue-green",
            "config": "blue_green_challenger_100",
            "champion_pct": 0,
            "challenger_pct": 100,
            "criterio_siguiente": "Promocion completa",
        },
        {
            "fase": 7,
            "nombre": "Rollback",
            "config": "rollback_champion_100",
            "champion_pct": 100,
            "challenger_pct": 0,
            "criterio_siguiente": "Usar si un guardrail falla",
        },
    ]
)

with iniciar_run("rollout_plan"):
    registrar_texto(rollout_plan.to_json(orient="records", indent=2), "rollout_plan.json")
    registrar_texto(json.dumps(serving_configs, indent=2), "serving_configs_dry_run.json")

mostrar(rollout_plan)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Despliegue real opcional
# MAGIC
# MAGIC La siguiente celda esta desactivada por defecto. Si quieres convertir esta demo en una demo real de Mosaic AI Model Serving, primero registra los modelos en Unity Catalog, actualiza `REGISTERED_MODEL_NAME`, confirma las versiones y cambia `ACTIVAR_DESPLIEGUE_REAL` a `True`.
# MAGIC
# MAGIC Con el valor por defecto, `Run all` solo imprime el payload que se enviaria al endpoint.

# COMMAND ----------

ACTIVAR_DESPLIEGUE_REAL = False
CONFIG_A_DESPLEGAR = "canary_5"

config_endpoint = serving_configs[CONFIG_A_DESPLEGAR]

if ACTIVAR_DESPLIEGUE_REAL:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import EndpointCoreConfigInput

    workspace_client = WorkspaceClient()
    endpoint_config = EndpointCoreConfigInput.from_dict(config_endpoint)

    workspace_client.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=endpoint_config,
    )
    print(f"Endpoint creado o actualizado: {ENDPOINT_NAME}")
else:
    print("Dry-run activado. No se creo ni actualizo ningun endpoint.")
    print(f"Endpoint objetivo: {ENDPOINT_NAME}")
    print(f"Configuracion seleccionada: {CONFIG_A_DESPLEGAR}")
    print(json.dumps(config_endpoint, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Limpieza opcional
# MAGIC
# MAGIC Esta celda tambien esta desactivada por defecto. Solo deberias activarla si creaste un endpoint real durante la demo.

# COMMAND ----------

ELIMINAR_ENDPOINT_REAL = False

if ACTIVAR_DESPLIEGUE_REAL and ELIMINAR_ENDPOINT_REAL:
    from databricks.sdk import WorkspaceClient

    workspace_client = WorkspaceClient()
    workspace_client.serving_endpoints.delete(name=ENDPOINT_NAME)
    print(f"Endpoint eliminado: {ENDPOINT_NAME}")
else:
    print("No se elimino ningun endpoint.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusiones
# MAGIC
# MAGIC Esta demo separa el concepto de rollout de la infraestructura real para que puedas explicar el ciclo completo sin riesgo operativo:
# MAGIC
# MAGIC - `champion_100` mantiene estabilidad.
# MAGIC - `shadow` permite comparar sin afectar usuarios.
# MAGIC - `canary` reduce blast radius.
# MAGIC - `A/B` permite comparar impacto con trafico balanceado.
# MAGIC - `blue-green` completa el cambio.
# MAGIC - `rollback` vuelve al champion si falla un guardrail.
# MAGIC
# MAGIC En un flujo productivo, estas fases se conectarian con modelos registrados en Unity Catalog, endpoints de Mosaic AI Model Serving, MLflow para trazabilidad y monitoreo continuo de calidad, latencia, drift y costo.

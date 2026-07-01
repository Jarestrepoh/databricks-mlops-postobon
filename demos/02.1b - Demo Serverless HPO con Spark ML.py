# Databricks notebook source
# MAGIC %md
# MAGIC # Demo autocontenida: HPO con Spark ML en Databricks Serverless
# MAGIC
# MAGIC Este notebook esta pensado para una demo de MLOps en Databricks usando Serverless Compute. La idea es mantener el espiritu del notebook original de Optuna, Hyperopt y Spark ML, pero sin depender de objetos de Databricks Academy, tablas externas, rutas del curso ni notebooks de setup.
# MAGIC
# MAGIC En esta demo vamos a:
# MAGIC
# MAGIC - Crear un dataset sintetico con forma de datos de calidad de vino.
# MAGIC - Entrenar un modelo base con Spark ML.
# MAGIC - Optimizar hiperparametros con Optuna mientras el entrenamiento se ejecuta con Spark ML.
# MAGIC - Comparar con `CrossValidator`, la opcion nativa de Spark ML.
# MAGIC - Usar Hyperopt con `Trials()` para mostrar una alternativa historica de busqueda.
# MAGIC - Registrar metricas y parametros en MLflow cuando este disponible.
# MAGIC
# MAGIC > Nota: Hyperopt ya no se recomienda para nuevas implementaciones en Databricks. Se incluye aqui con fines comparativos porque muchas organizaciones todavia tienen notebooks o pipelines historicos que lo usan.

# COMMAND ----------

# MAGIC %pip install -q -U optuna optuna-integration hyperopt

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuracion de la demo
# MAGIC
# MAGIC La demo evita dependencias de datos externos. El unico supuesto es estar en un notebook de Databricks con Spark disponible.
# MAGIC
# MAGIC MLflow se configura usando el directorio del notebook actual. Si MLflow no esta disponible, el notebook continua y muestra los resultados en pantalla.

# COMMAND ----------

import json
import os
import time
from contextlib import nullcontext

from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import (
    DecisionTreeRegressor,
    GBTRegressor,
    LinearRegression,
    RandomForestRegressor,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

SEED = 42
NUM_FILAS = 6000
LABEL_COL = "quality"
FEATURE_COLS = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def mostrar(df, n=10):
    """Muestra un DataFrame en Databricks o en consola si display no existe."""
    try:
        display(df.limit(n))
    except Exception:
        df.show(n, truncate=False)


def obtener_directorio_notebook(default="/Shared"):
    """Obtiene el directorio del notebook sin depender de rutas del curso."""
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
EXPERIMENT_NAME = f"{obtener_directorio_notebook()}/02.1b - Demo Serverless HPO con Spark ML"

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


evaluator = RegressionEvaluator(
    labelCol=LABEL_COL,
    predictionCol="prediction",
    metricName="rmse",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Crear un dataset sintetico de calidad de vino
# MAGIC
# MAGIC Para que el notebook sea autocontenido, generamos los datos con Spark. Las columnas conservan nombres similares al dataset de calidad de vino usado en muchos ejemplos de Spark ML.
# MAGIC
# MAGIC La variable objetivo `quality` se construye con una relacion controlada: mayor alcohol, sulfitos y acidez citrica tienden a subir la calidad; mayor acidez volatil, cloruros y dioxido de azufre tienden a bajarla. Esto permite que los modelos aprendan un patron realista sin leer ningun archivo externo.

# COMMAND ----------

wine_df = (
    spark.range(NUM_FILAS)
    .withColumn("fixed_acidity", F.round(F.lit(4.5) + F.rand(SEED + 1) * F.lit(8.0), 3))
    .withColumn("volatile_acidity", F.round(F.lit(0.10) + F.rand(SEED + 2) * F.lit(1.10), 3))
    .withColumn("citric_acid", F.round(F.rand(SEED + 3) * F.lit(0.80), 3))
    .withColumn("residual_sugar", F.round(F.lit(1.0) + F.rand(SEED + 4) * F.lit(11.0), 3))
    .withColumn("chlorides", F.round(F.lit(0.030) + F.rand(SEED + 5) * F.lit(0.150), 4))
    .withColumn("free_sulfur_dioxide", F.round(F.lit(3.0) + F.rand(SEED + 6) * F.lit(62.0), 3))
    .withColumn("total_sulfur_dioxide", F.round(F.lit(10.0) + F.rand(SEED + 7) * F.lit(170.0), 3))
    .withColumn("density", F.round(F.lit(0.9900) + F.rand(SEED + 8) * F.lit(0.0120), 5))
    .withColumn("pH", F.round(F.lit(2.90) + F.rand(SEED + 9) * F.lit(0.80), 3))
    .withColumn("sulphates", F.round(F.lit(0.30) + F.rand(SEED + 10) * F.lit(0.90), 3))
    .withColumn("alcohol", F.round(F.lit(8.0) + F.rand(SEED + 11) * F.lit(6.0), 3))
)

quality_signal = (
    F.lit(4.2)
    + F.col("alcohol") * F.lit(0.28)
    + F.col("sulphates") * F.lit(0.90)
    + F.col("citric_acid") * F.lit(0.60)
    - F.col("volatile_acidity") * F.lit(1.80)
    - F.col("chlorides") * F.lit(6.00)
    - F.col("residual_sugar") * F.lit(0.025)
    - F.col("total_sulfur_dioxide") * F.lit(0.003)
    + F.randn(SEED + 12) * F.lit(0.25)
)

wine_df = (
    wine_df.withColumn(
        LABEL_COL,
        F.round(F.least(F.lit(9.0), F.greatest(F.lit(3.0), quality_signal)), 2).cast("double"),
    )
    .drop("id")
    # .cache()
)

print(f"Filas generadas: {wine_df.count():,}")
mostrar(wine_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Preparar features y dividir train/test
# MAGIC
# MAGIC Spark ML espera una columna vectorial `features`. Usamos `VectorAssembler`, dividimos los datos y cacheamos los subconjuntos para que las busquedas de hiperparametros reutilicen el mismo material de entrenamiento.

# COMMAND ----------

assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")

model_df = assembler.transform(wine_df).select(*FEATURE_COLS, "features", LABEL_COL)
train_df, test_df = model_df.randomSplit([0.8, 0.2], seed=SEED)
train_df = train_df
test_df = test_df

train_count = train_df.count()
test_count = test_df.count()

print(f"Filas de entrenamiento: {train_count:,}")
print(f"Filas de prueba:        {test_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Baseline: Random Forest sin tuning
# MAGIC
# MAGIC Antes de optimizar, entrenamos un modelo base. Este valor sera la referencia para saber si las estrategias de HPO mejoran el resultado.

# COMMAND ----------

with iniciar_run("baseline_random_forest"):
    inicio = time.time()

    baseline_rf = RandomForestRegressor(
        featuresCol="features",
        labelCol=LABEL_COL,
        numTrees=20,
        maxDepth=5,
        seed=SEED,
    )
    baseline_model = baseline_rf.fit(train_df)
    baseline_rmse = evaluator.evaluate(baseline_model.transform(test_df))
    baseline_duration = time.time() - inicio

    registrar_parametros(
        {
            "modelo": "RandomForestRegressor",
            "numTrees": 20,
            "maxDepth": 5,
            "seed": SEED,
        }
    )
    registrar_metricas(
        {
            "test_rmse": baseline_rmse,
            "duracion_s": baseline_duration,
        }
    )

print(f"Baseline RMSE: {baseline_rmse:.4f}")
print(f"Duracion baseline: {baseline_duration:.1f} segundos")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Optuna + Spark ML
# MAGIC
# MAGIC Optuna administra la busqueda en el driver. Cada configuracion entrena un modelo de Spark ML, por lo que el entrenamiento aprovecha la ejecucion distribuida de Spark.
# MAGIC
# MAGIC En Serverless conviene mantener esta parte simple y reproducible: ejecutamos pocos ensayos y evitamos paralelizar varios entrenamientos Spark desde el mismo notebook.

# COMMAND ----------

import optuna


class ObjetivoOptuna:
    """Funcion objetivo para comparar varios modelos de Spark ML con Optuna."""

    def __init__(self, train_data, test_data, evaluator):
        self.train_data = train_data
        self.test_data = test_data
        self.evaluator = evaluator

    def __call__(self, trial):
        modelo = trial.suggest_categorical(
            "modelo",
            ["LinearRegression", "RandomForestRegressor", "GBTRegressor"],
        )

        if modelo == "LinearRegression":
            estimator = LinearRegression(
                featuresCol="features",
                labelCol=LABEL_COL,
                maxIter=50,
                regParam=trial.suggest_float("lr_reg_param", 0.001, 0.50, log=True),
                elasticNetParam=trial.suggest_float("lr_elastic_net_param", 0.0, 1.0),
            )
        elif modelo == "RandomForestRegressor":
            estimator = RandomForestRegressor(
                featuresCol="features",
                labelCol=LABEL_COL,
                seed=SEED,
                numTrees=trial.suggest_int("rf_num_trees", 10, 40, step=10),
                maxDepth=trial.suggest_int("rf_max_depth", 3, 8),
                minInstancesPerNode=trial.suggest_int("rf_min_instances_per_node", 1, 5),
            )
        else:
            estimator = GBTRegressor(
                featuresCol="features",
                labelCol=LABEL_COL,
                seed=SEED,
                maxDepth=trial.suggest_int("gbt_max_depth", 2, 6),
                maxIter=trial.suggest_int("gbt_max_iter", 10, 30, step=10),
                stepSize=trial.suggest_float("gbt_step_size", 0.03, 0.30),
            )

        modelo_entrenado = estimator.fit(self.train_data)
        predicciones = modelo_entrenado.transform(self.test_data)
        rmse = self.evaluator.evaluate(predicciones)
        trial.set_user_attr("rmse", rmse)
        return rmse


def construir_modelo_optuna(params):
    modelo = params["modelo"]

    if modelo == "LinearRegression":
        return LinearRegression(
            featuresCol="features",
            labelCol=LABEL_COL,
            maxIter=50,
            regParam=float(params["lr_reg_param"]),
            elasticNetParam=float(params["lr_elastic_net_param"]),
        )

    if modelo == "RandomForestRegressor":
        return RandomForestRegressor(
            featuresCol="features",
            labelCol=LABEL_COL,
            seed=SEED,
            numTrees=int(params["rf_num_trees"]),
            maxDepth=int(params["rf_max_depth"]),
            minInstancesPerNode=int(params["rf_min_instances_per_node"]),
        )

    return GBTRegressor(
        featuresCol="features",
        labelCol=LABEL_COL,
        seed=SEED,
        maxDepth=int(params["gbt_max_depth"]),
        maxIter=int(params["gbt_max_iter"]),
        stepSize=float(params["gbt_step_size"]),
    )


optuna.logging.set_verbosity(optuna.logging.WARNING)

sampler = optuna.samplers.TPESampler(
    n_startup_trials=4,
    seed=SEED,
)

study = optuna.create_study(
    study_name="serverless_spark_ml_hpo",
    sampler=sampler,
    direction="minimize",
)

N_TRIALS_OPTUNA = 12
objetivo = ObjetivoOptuna(train_df, test_df, evaluator)

with iniciar_run("optuna_spark_ml"):
    inicio = time.time()
    study.optimize(objetivo, n_trials=N_TRIALS_OPTUNA, n_jobs=1)
    optuna_duration = time.time() - inicio

    best_optuna_params = study.best_trial.params
    optuna_final_model = construir_modelo_optuna(best_optuna_params).fit(train_df)
    optuna_rmse = evaluator.evaluate(optuna_final_model.transform(test_df))

    registrar_parametros(
        {
            "buscador": "Optuna",
            "n_trials": N_TRIALS_OPTUNA,
            "mejor_modelo": best_optuna_params["modelo"],
            **best_optuna_params,
        }
    )
    registrar_metricas(
        {
            "best_trial_rmse": study.best_trial.value,
            "test_rmse": optuna_rmse,
            "duracion_s": optuna_duration,
        }
    )

    resumen_optuna = [
        {
            "numero": trial.number,
            "estado": str(trial.state),
            "rmse": trial.value,
            "parametros": trial.params,
        }
        for trial in study.trials
    ]
    registrar_texto(json.dumps(resumen_optuna, indent=2), "optuna_trials.json")

print(f"Mejor ensayo Optuna: {study.best_trial.number}")
print(f"Mejores parametros Optuna: {best_optuna_params}")
print(f"RMSE final Optuna: {optuna_rmse:.4f}")
print(f"Duracion Optuna: {optuna_duration:.1f} segundos")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Tuning nativo con Spark ML CrossValidator
# MAGIC
# MAGIC `CrossValidator` es la opcion mas integrada para tuning de modelos Spark ML. La busqueda no es tan flexible como Optuna, pero Spark administra el ajuste y puede evaluar combinaciones con paralelismo controlado.

# COMMAND ----------

os.environ["SPARKML_TEMP_DFS_PATH"] = "/Volumes/mlops_dbx_talk_dev/ezapata/helpers/pyspark_ml_models/hpo_demo"

rf_cv = RandomForestRegressor(
    featuresCol="features",
    labelCol=LABEL_COL,
    seed=SEED,
)

param_grid = (
    ParamGridBuilder()
    .addGrid(rf_cv.numTrees, [10, 20, 40])
    .addGrid(rf_cv.maxDepth, [3, 5, 8])
    .addGrid(rf_cv.minInstancesPerNode, [1, 3])
    .build()
)

cross_validator = CrossValidator(
    estimator=rf_cv,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=2,
    seed=SEED,
)

with iniciar_run("spark_ml_cross_validator"):
    inicio = time.time()
    cv_model = cross_validator.fit(train_df)
    cv_duration = time.time() - inicio

    best_cv_model = cv_model.bestModel
    cv_rmse = evaluator.evaluate(best_cv_model.transform(test_df))

    best_cv_params = {
        "numTrees": best_cv_model.getNumTrees,
        "maxDepth": best_cv_model.getOrDefault(best_cv_model.maxDepth),
        "minInstancesPerNode": best_cv_model.getOrDefault(best_cv_model.minInstancesPerNode),
    }

    resultados_cv = []
    for indice, param_map in enumerate(param_grid):
        resultados_cv.append(
            {
                "numTrees": param_map[rf_cv.numTrees],
                "maxDepth": param_map[rf_cv.maxDepth],
                "minInstancesPerNode": param_map[rf_cv.minInstancesPerNode],
                "avg_rmse": float(cv_model.avgMetrics[indice]),
            }
        )

    registrar_parametros(
        {
            "buscador": "Spark ML CrossValidator",
            "numFolds": 3,
            "parallelism": 2,
            **best_cv_params,
        }
    )
    registrar_metricas(
        {
            "test_rmse": cv_rmse,
            "duracion_s": cv_duration,
        }
    )
    registrar_texto(json.dumps(resultados_cv, indent=2), "spark_cv_resultados.json")

print(f"Mejores parametros Spark CV: {best_cv_params}")
print(f"RMSE final Spark CV: {cv_rmse:.4f}")
print(f"Duracion Spark CV: {cv_duration:.1f} segundos")

mostrar(spark.createDataFrame(resultados_cv).orderBy("avg_rmse"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Hyperopt + Spark ML
# MAGIC
# MAGIC Hyperopt se muestra como referencia para notebooks existentes. En esta version usamos `Trials()` y dejamos que cada evaluacion llame a Spark ML desde el driver. Asi evitamos pasar objetos de Spark a procesos externos.
# MAGIC
# MAGIC El modelo usado aqui es un arbol de decision para que la seccion sea rapida y facil de explicar durante una charla.

# COMMAND ----------

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


def entrenar_arbol_decision(max_depth, max_bins, min_instances_per_node):
    estimator = DecisionTreeRegressor(
        featuresCol="features",
        labelCol=LABEL_COL,
        seed=SEED,
        maxDepth=int(max_depth),
        maxBins=int(max_bins),
        minInstancesPerNode=int(min_instances_per_node),
    )
    modelo = estimator.fit(train_df)
    rmse = evaluator.evaluate(modelo.transform(test_df))
    return modelo, rmse


def objetivo_hyperopt(params):
    max_depth = int(params["maxDepth"])
    max_bins = int(params["maxBins"])
    min_instances = int(params["minInstancesPerNode"])

    _, rmse = entrenar_arbol_decision(
        max_depth=max_depth,
        max_bins=max_bins,
        min_instances_per_node=min_instances,
    )

    return {
        "loss": rmse,
        "status": STATUS_OK,
        "maxDepth": max_depth,
        "maxBins": max_bins,
        "minInstancesPerNode": min_instances,
    }


space = {
    "maxDepth": hp.quniform("maxDepth", 2, 8, 1),
    "maxBins": hp.quniform("maxBins", 16, 64, 1),
    "minInstancesPerNode": hp.quniform("minInstancesPerNode", 1, 8, 1),
}

N_TRIALS_HYPEROPT = 12
trials = Trials()

with iniciar_run("hyperopt_spark_ml"):
    inicio = time.time()
    best_hyperopt_raw = fmin(
        fn=objetivo_hyperopt,
        space=space,
        algo=tpe.suggest,
        max_evals=N_TRIALS_HYPEROPT,
        trials=trials,
        rstate=np.random.default_rng(SEED),
    )
    hyperopt_duration = time.time() - inicio

    best_hyperopt_params = {
        "maxDepth": int(best_hyperopt_raw["maxDepth"]),
        "maxBins": int(best_hyperopt_raw["maxBins"]),
        "minInstancesPerNode": int(best_hyperopt_raw["minInstancesPerNode"]),
    }

    hyperopt_model, hyperopt_rmse = entrenar_arbol_decision(
        max_depth=best_hyperopt_params["maxDepth"],
        max_bins=best_hyperopt_params["maxBins"],
        min_instances_per_node=best_hyperopt_params["minInstancesPerNode"],
    )

    resumen_hyperopt = []
    for indice, trial in enumerate(trials.trials):
        resultado = trial["result"]
        resumen_hyperopt.append(
            {
                "numero": indice,
                "rmse": resultado.get("loss"),
                "maxDepth": resultado.get("maxDepth"),
                "maxBins": resultado.get("maxBins"),
                "minInstancesPerNode": resultado.get("minInstancesPerNode"),
            }
        )

    registrar_parametros(
        {
            "buscador": "Hyperopt",
            "n_trials": N_TRIALS_HYPEROPT,
            **best_hyperopt_params,
        }
    )
    registrar_metricas(
        {
            "test_rmse": hyperopt_rmse,
            "duracion_s": hyperopt_duration,
        }
    )
    registrar_texto(json.dumps(resumen_hyperopt, indent=2), "hyperopt_trials.json")

print(f"Mejores parametros Hyperopt: {best_hyperopt_params}")
print(f"RMSE final Hyperopt: {hyperopt_rmse:.4f}")
print(f"Duracion Hyperopt: {hyperopt_duration:.1f} segundos")

mostrar(spark.createDataFrame(resumen_hyperopt).orderBy("rmse"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Comparacion final
# MAGIC
# MAGIC La tabla final resume el RMSE de cada enfoque. En una demo real, este cierre permite conectar el resultado tecnico con una conversacion de MLOps:
# MAGIC
# MAGIC - Todos los enfoques usan el mismo split y la misma metrica.
# MAGIC - MLflow permite auditar que parametros produjo cada busqueda.
# MAGIC - El notebook puede ejecutarse de punta a punta en un entorno Serverless sin preparar tablas ni archivos.

# COMMAND ----------

comparacion = [
    ("Baseline Random Forest", float(baseline_rmse), float(baseline_duration)),
    ("Optuna + Spark ML", float(optuna_rmse), float(optuna_duration)),
    ("Spark ML CrossValidator", float(cv_rmse), float(cv_duration)),
    ("Hyperopt + Spark ML", float(hyperopt_rmse), float(hyperopt_duration)),
]

comparacion_df = spark.createDataFrame(
    comparacion,
    ["enfoque", "rmse", "duracion_s"],
).orderBy("rmse")

with iniciar_run("resumen_comparativo"):
    mejor = comparacion_df.first()
    registrar_parametros({"mejor_enfoque": mejor["enfoque"]})
    registrar_metricas({"mejor_rmse": mejor["rmse"]})
    registrar_texto(
        json.dumps(
            [
                {
                    "enfoque": fila["enfoque"],
                    "rmse": fila["rmse"],
                    "duracion_s": fila["duracion_s"],
                }
                for fila in comparacion_df.collect()
            ],
            indent=2,
        ),
        "comparacion_final.json",
    )

mostrar(comparacion_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusiones
# MAGIC
# MAGIC En esta demo vimos tres formas de hacer tuning de hiperparametros con Spark ML en Databricks:
# MAGIC
# MAGIC - **Optuna**: flexible para definir espacios de busqueda y comparar familias de modelos.
# MAGIC - **Spark ML CrossValidator**: opcion nativa, integrada y directa para modelos Spark ML.
# MAGIC - **Hyperopt**: util como referencia para cargas existentes, aunque no recomendado para nuevos desarrollos.
# MAGIC
# MAGIC Para una estrategia moderna en Databricks, una practica comun es combinar Spark ML para entrenamiento distribuido, Optuna o herramientas nativas para busqueda controlada, y MLflow para trazabilidad de experimentos.
# MAGIC

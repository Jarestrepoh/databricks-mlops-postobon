# Architecture: End-to-End Databricks MLOps Workshop Project

## 1. Purpose and Scope
This architecture describes the end-to-end flow for a telco churn ML project used in a workshop/talk setting. It focuses on clarity and reproducibility for ML engineers and data scientists.

## 2. High-Level System Flow
1. Data ingestion/load creates raw train and inference tables.
2. Feature engineering workflow computes customer-level features and writes a feature table.
3. Model workflow trains, validates, and deploys a model alias.
4. Batch inference workflow scores new records and writes predictions.
5. Monitoring evaluates inference quality and conditionally triggers retraining.

## 3. Core Components
Code and notebooks:
- Feature engineering notebooks and module:
  - mlops_dbx/feature_engineering/notebooks/LoadData.ipynb
  - mlops_dbx/feature_engineering/notebooks/GenerateAndWriteFeatures.ipynb
  - mlops_dbx/feature_engineering/features/compute_features.py
- Training notebooks:
  - mlops_dbx/training/notebooks/TrainWithFeatureStore.ipynb
  - mlops_dbx/training/notebooks/TrainWithFeatureStore_Spark.ipynb
- Validation:
  - mlops_dbx/validation/notebooks/ModelValidation.ipynb
  - mlops_dbx/validation/validation.py
- Deployment:
  - mlops_dbx/deployment/model_deployment/notebooks/ModelDeployment.ipynb
  - mlops_dbx/deployment/model_deployment/deploy.py
- Batch inference:
  - mlops_dbx/deployment/batch_inference/notebooks/BatchInference.ipynb
  - mlops_dbx/deployment/batch_inference/predict.py
- Monitoring:
  - mlops_dbx/monitoring/notebooks/MonitoredMetricViolationCheck.ipynb
  - mlops_dbx/monitoring/metric_violation_check_query.py

Resource configuration:
- Bundle root: mlops_dbx/databricks.yml
- Workflow resources under: mlops_dbx/resources

## 4. Data and Model Contracts
Data platform assumptions:
- Unity Catalog managed tables
- Delta format for persisted inputs/outputs

Primary table contracts (by configuration):
- Training raw: ${catalog}.churn.telco_churn_train_raw
- Feature table: ${catalog}.churn.telco_cust_features
- Labels: ${catalog}.churn.telco_cust_labels
- Validation input: ${catalog}.churn.telco_churn_validation
- Batch inference input: ${catalog}.churn.telco_churn_inference_raw
- Prediction output: ${catalog}.churn.telco_churn_predictions
- Monitored inference table: ${catalog}.churn.telco_churn_inference_table

Model registry contract:
- Registry URI: databricks-uc
- Model name pattern: ${catalog}.churn.telco_churn_model
- Deployment alias strategy: assign champion alias

## 5. Orchestration and Scheduling
Feature engineering job:
- Resource: feature-engineering-workflow-resource.yml
- Scheduled daily around 07:00 UTC

Monitoring/retraining job:
- Monitor schedule around 08:00 UTC
- Retraining orchestration schedule around 18:00 UTC

Model training/validation/deployment job:
- Resource: model-workflow-resource.yml
- Scheduled daily around 09:00 UTC

Batch inference job:
- Resource: batch-inference-workflow-resource.yml
- Scheduled daily around 11:00 UTC

## 6. Environment Topology
Bundle targets:
- dev (development mode, default)
- staging
- prod
- test

Environment-specific behavior:
- Catalog variable per target: mlops_dbx_talk_${bundle.target}
- Workspace paths vary by target root_path

## 7. CI/CD Architecture
PR stage:
- Run unit tests (pytest + local Spark)
- Validate bundle for staging/prod
- Deploy and run integration jobs on test target in staging workspace

Merge to main:
- Validate and deploy to staging target

Merge to release:
- Validate and deploy to prod target

## 8. Failure Domains and Recovery Strategy
Common failure domains:
- Data schema drift or missing required columns in feature pipeline
- Validation metric threshold failure
- Bundle deployment authentication/config issues
- Monitoring false positives/negatives due to label lag

Recovery strategies:
- Re-run failed Databricks job task from failed step after correction
- Adjust validation thresholds and rerun training workflow
- Roll back deployed alias to previous stable model version
- Disable retraining trigger temporarily during incident triage

## 9. Scalability Notes
- Feature logic is implemented with Spark DataFrame transformations and is cluster-scalable.
- Batch scoring uses feature engineering client score_batch for distributed inference.
- Avoid single-node data pulls in production-scale runs.

## 10. Assumptions and Open Items
Assumptions:
- Demo datasets and UC objects exist per environment.
- Required secrets and service principal credentials are configured in GitHub.

Open items to harden before production:
- Resolve remaining TODO markers in resource files.
- Define explicit ownership and escalation contacts.
- Add formal SLOs and richer model quality gates.

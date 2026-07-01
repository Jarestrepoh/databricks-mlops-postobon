# Model Card: Telco Customer Churn Classifier

## 1. Model Details
- Model name pattern: ${catalog}.churn.telco_churn_model
- Registry: Unity Catalog via MLflow (databricks-uc)
- Problem type: Binary classification (churn vs no churn)
- Frameworks and stack:
  - PySpark for feature engineering
  - MLflow for tracking, registry, and validation
  - Databricks workflows for orchestration

## 2. Intended Use
Intended uses:
- Workshop demonstration of end-to-end MLOps on Databricks
- Educational examples for ML engineers and data scientists
- Batch churn risk scoring in a controlled demo setup

Out-of-scope uses:
- Direct production decisions affecting customers without domain and legal review
- High-stakes decisions in regulated domains without additional controls and governance
- Real-time serving SLO commitments (current reference flow is batch-oriented)

## 3. Data
Primary entities:
- Customer-level telco records keyed by customerID (mapped to customer_id)

Data assets referenced by workflows:
- Training raw table: ${catalog}.churn.telco_churn_train_raw
- Validation table: ${catalog}.churn.telco_churn_validation
- Inference raw table: ${catalog}.churn.telco_churn_inference_raw
- Feature table: ${catalog}.churn.telco_cust_features
- Label table: ${catalog}.churn.telco_cust_labels
- Inference monitored table: ${catalog}.churn.telco_churn_inference_table

Data characteristics:
- Mix of categorical service attributes, contract/payment metadata, and billing/tenure signals
- Feature pipeline excludes target label from feature table to reduce leakage risk

## 4. Features
Feature engineering implementation:
- Source: mlops_dbx/feature_engineering/features/compute_features.py

Notable engineered features:
- Household and service flags (partner, dependents, internet/add-on services)
- Contract and payment indicators (month-to-month, automatic payment)
- Tenure buckets and billing consistency signals
- Missingness indicator and robust numeric casting for TotalCharges

Feature table contract:
- Primary key: customer_id
- Label is intentionally excluded from output feature table

## 5. Training and Evaluation
Workflow:
- Training notebook executed via model workflow resource
- Validation notebook executed after training

Validation configuration source:
- mlops_dbx/validation/validation.py

Current configured custom metric and threshold:
- Custom metric: recall
- Threshold: recall >= 0.5
- Baseline comparison: disabled in current resource config

Evaluation caveat:
- A single threshold is used for workshop demonstration. Production usage should define richer threshold sets and regression checks.

## 6. Ethical and Operational Considerations
Potential risks:
- Data drift in customer behavior or service mix
- Label latency and incomplete ground truth for monitoring windows
- Bias risk across customer cohorts if slicing is not monitored
- Misuse risk if scores are interpreted as deterministic outcomes

Mitigations in current design:
- Monitoring job with metric violation checks over recent windows
- Retraining trigger workflow integrated with monitor condition task
- Explicit runbook guidance for incident triage and rollback

Recommended enhancements:
- Add cohort-level fairness slices and parity checks
- Add calibration and precision-recall threshold policy by business segment
- Add data quality expectation checks before training and scoring

## 7. Monitoring and Retraining
Monitoring resource:
- mlops_dbx/resources/monitoring-resource.yml

Current monitor setup:
- Monitor type: inference log quality monitor
- Metric under violation check: accuracy_score
- Violation logic: threshold breaches in recent windows plus latest-window breach
- Trigger action: run model_training_job when condition is true

## 8. Governance
Versioning and traceability:
- Model versions tracked in Unity Catalog Model Registry
- Git source metadata passed as workflow parameters for run traceability

Ownership model (workshop default):
- ML owner: workshop facilitator or assigned ML engineer
- Platform owner: Databricks workspace administrator
- Incident responder: on-call ML/platform role

## 9. Limitations
- Workshop-first defaults may not represent production-safe thresholds
- Some template TODO markers remain in resource files and should be resolved before production use
- Monitoring efficacy depends on label timeliness and quality

## 10. Approval Status
- Current status: Workshop-ready demonstration artifact
- Production status: Not approved without environment-specific risk and compliance review

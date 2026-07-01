# Databricks MLOps Talk: End-to-End Telco Churn ML Project

This repository showcases an end-to-end ML lifecycle on Databricks for a workshop/talk audience.

Primary audience:
- ML engineers
- Data scientists

Project goal:
- Demonstrate how to move from feature engineering to training, validation, deployment, batch inference, and monitoring/retraining with Databricks Asset Bundles, Unity Catalog, and MLflow.

## Documentation Map
- Model card: [docs/model-card.md](docs/model-card.md)
- Architecture: [docs/architecture.md](docs/architecture.md)
- Operations runbook: [docs/runbook.md](docs/runbook.md)

Legacy Databricks MLOps Stacks docs were preserved with the suffix _mlopstacks.

## Repository Structure
- Core ML project: mlops_dbx
- CI/CD workflows: .github/workflows
- Workshop docs: docs

Key modules:
- Feature engineering logic: mlops_dbx/feature_engineering/features/compute_features.py
- Validation thresholds and custom metrics: mlops_dbx/validation/validation.py
- Batch inference scoring helper: mlops_dbx/deployment/batch_inference/predict.py
- Model deployment aliasing: mlops_dbx/deployment/model_deployment/deploy.py
- Monitoring metric violation SQL template: mlops_dbx/monitoring/metric_violation_check_query.py

## ML Lifecycle Implemented
1. Load and prepare raw telco data into Unity Catalog tables.
2. Generate customer features with Spark and write feature tables.
3. Train and register a churn model in Unity Catalog Model Registry.
4. Validate model quality (custom recall threshold currently configured).
5. Deploy by assigning model alias (champion).
6. Run batch inference and write predictions to Delta tables.
7. Monitor inference quality and trigger retraining when threshold conditions are met.

## Databricks Workflows in Bundle Resources
- Feature engineering workflow: mlops_dbx/resources/feature-engineering-workflow-resource.yml
- Model workflow (train, validate, deploy): mlops_dbx/resources/model-workflow-resource.yml
- Batch inference workflow: mlops_dbx/resources/batch-inference-workflow-resource.yml
- Monitoring and retraining workflow: mlops_dbx/resources/monitoring-resource.yml
- Model registry artifacts: mlops_dbx/resources/ml-artifacts-resource.yml

## Deployment Targets
Bundle targets in mlops_dbx/databricks.yml:
- dev (default)
- staging
- prod
- test

Current catalog variable pattern:
- mlops_dbx_talk_${bundle.target}

## CI/CD Summary
- PR checks:
  - Unit tests (pytest)
  - Bundle validate for staging and prod
  - Integration deployment and workflow runs against test target in staging workspace
- Main branch:
  - Deploy to staging
- Release branch:
  - Deploy to prod

See workflows in .github/workflows for exact behavior.

## Quickstart
1. Install dependencies:
   - pip install -r mlops_dbx/requirements.txt
   - pip install -r test-requirements.txt
2. Run unit tests:
   - cd mlops_dbx
   - pytest
3. Validate Databricks bundle:
   - databricks bundle validate -t dev
4. Deploy bundle:
   - databricks bundle deploy -t dev

## Known Workshop Constraints
- This project is optimized for educational demonstration, not strict production hardening.
- Notebook-driven orchestration is intentionally transparent for teaching.
- Some resource files still include TODO comments from template origins.
- Monitoring thresholds and runbook severities should be tuned per real business context.

## Preserved Legacy Docs
Original stack docs were renamed and retained, for example:
- README_mlopstacks.md
- docs/mlops-setup_mlopstacks.md
- mlops_dbx/README_mlopstacks.md
- mlops_dbx/resources/README_mlopstacks.md

# Operations Runbook: Databricks MLOps Workshop Project

## 1. Purpose
This runbook provides operational procedures for feature, training, deployment, batch inference, and monitoring workflows in this repository.

Default audience:
- ML engineers
- Data scientists supporting MLOps operations

## 2. Operational Scope
In-scope components:
- Databricks bundle deployments
- Databricks jobs under mlops_dbx/resources
- Unity Catalog model alias deployment
- Monitoring-triggered retraining workflow

Out-of-scope:
- Non-Databricks infrastructure incidents
- Regulated production governance workflows not configured in this repository

## 3. Service Inventory
Primary jobs:
- write_feature_table_job
- model_training_job
- batch_inference_job
- retraining_job

Critical resources:
- Bundle config: mlops_dbx/databricks.yml
- Workflow configs: mlops_dbx/resources/*.yml
- Validation thresholds: mlops_dbx/validation/validation.py

## 4. Severity Model
- SEV-1: Production scoring unavailable or corrupted outputs in prod target.
- SEV-2: Staging/prod training/deployment blocked, no immediate customer impact.
- SEV-3: Dev/test workflow failures, documentation/config drift, non-blocking defects.

## 5. Alert and Triage Workflow
1. Identify failing target and workflow from Databricks job run history.
2. Confirm latest code version and bundle target used in run parameters.
3. Classify severity based on impact and environment.
4. Follow scenario-specific playbook below.
5. Record incident notes and recovery action.

## 6. Scenario Playbooks
### A. Feature Job Failure
Symptoms:
- write_feature_table_job fails in LoadData or GenerateFeatures.

Checks:
- Validate required input table exists and schema matches compute_features expectations.
- Check for missing columns listed in compute_features.py REQUIRED_COLUMNS.
- Verify catalog/schema permissions.

Actions:
1. Fix schema or upstream data issue.
2. Re-run write_feature_table_job for impacted target.
3. Confirm output feature table refresh succeeded.

Exit criteria:
- Feature table updated and downstream training can execute.

### B. Model Validation Failure
Symptoms:
- ModelValidation task fails threshold checks (for example recall < 0.5).

Checks:
- Inspect MLflow evaluation outputs.
- Confirm validation input table freshness and label availability.
- Confirm threshold definitions in validation.py.

Actions:
1. Decide if failure is expected regression or data issue.
2. If data issue, remediate data and retrain.
3. If threshold needs recalibration for workshop context, update validation.py and redeploy.
4. Re-run model_training_job.

Exit criteria:
- Validation passes for approved threshold policy.

### C. Deployment Alias Issues
Symptoms:
- ModelDeployment task succeeds but champion alias points to wrong version, or deployment task fails.

Checks:
- Confirm model URI and alias mapping logic in deploy.py.
- Verify registry permissions for alias mutation.

Actions:
1. Manually inspect registered model aliases in Unity Catalog.
2. Reassign champion alias to last known good version if required.
3. Re-run deployment task after permission/config fixes.

Exit criteria:
- Champion alias points to intended model version.

### D. Batch Inference Failure
Symptoms:
- batch_inference_job fails, or predictions table not refreshed.

Checks:
- Verify inference input table exists and has expected schema.
- Confirm model alias resolves and model is loadable.
- Confirm output table write permissions and target table path.

Actions:
1. Fix schema/access issues.
2. Re-run GenerateFeatures then batch_inference_job.
3. Validate output table row count and timestamp freshness.

Exit criteria:
- Predictions table updated with current model_id and timestamp.

### E. Monitoring Trigger Noise or Missed Retraining
Symptoms:
- Frequent false alerts or no retraining despite poor metrics.

Checks:
- Review metric violation logic in metric_violation_check_query.py.
- Verify monitor metric, operator, threshold, and windows in monitoring-resource.yml.
- Verify label join quality in monitored inference table.

Actions:
1. Tune violation threshold and window parameters.
2. Validate monitor data quality and label latency assumptions.
3. Re-run monitored_metric_violation_check task.

Exit criteria:
- Trigger behavior matches intended policy.

## 7. Rollback and Forward-Fix
Rollback options:
- Re-point champion alias to previous stable model version.
- Re-deploy previous known-good commit to staging/prod.

Forward-fix options:
- Patch workflow config and redeploy bundle.
- Update validation thresholds/metrics and retrain.
- Repair data contracts then rerun failed jobs.

Verification after rollback or fix:
1. Validate bundle for target.
2. Run affected workflow manually.
3. Confirm output table and model alias state.
4. Document incident closure notes.

## 8. Standard Commands
From mlops_dbx directory:
- databricks bundle validate -t <target>
- databricks bundle deploy -t <target>
- databricks bundle run write_feature_table_job -t <target>
- databricks bundle run model_training_job -t <target>
- databricks bundle run batch_inference_job -t <target>

Local test commands:
- pytest

## 9. Ownership and Escalation (Workshop Default)
- Primary ML owner: workshop facilitator or designated ML engineer
- Platform owner: Databricks workspace admin/support
- CI/CD owner: repository maintainer

Escalation path:
1. On-call ML engineer
2. Platform owner
3. Repository maintainer / workshop lead

## 10. Change Management and Review Cadence
- Review workflow/resource configuration changes on every PR touching mlops_dbx/resources or mlops_dbx/databricks.yml.
- Review model card and architecture docs at least once per workshop iteration.
- Record major threshold or deployment policy changes in PR descriptions.

# Azure DevOps CI/CD

This repo contains three Azure Pipeline definitions that replace the active GitHub Actions workflows:

| GitHub Actions workflow | Azure Pipeline YAML | Behavior |
| --- | --- | --- |
| `.github/workflows/mlops_dbx-run-tests.yml` | `azure-pipelines-ci.yml` | Unit tests, test bundle deploy, feature workflow run, training workflow run |
| `.github/workflows/mlops_dbx-bundle-ci.yml` | `azure-pipelines-ci.yml` | Bundle validation for `staging` and `prod` |
| `.github/workflows/mlops_dbx-bundle-cd-staging.yml` | `azure-pipelines-cd-staging.yml` | Deploys `staging` from `main` |
| `.github/workflows/mlops_dbx-bundle-cd-prod.yml` | `azure-pipelines-cd-prod.yml` | Deploys `prod` from `release` |

The old `deploy-cicd.yml` generated GitHub workflow files, so it is not needed after migrating the workflow definitions to Azure Pipelines. The old `lint-cicd-workflow-files.yml` only linted GitHub Actions YAML.

## Required Azure DevOps Setup

1. Push this repository to Azure Repos with `main` and `release` branches.
2. In Azure DevOps, open `Pipelines > Library`.
3. Create a variable group named `databricks-cicd`.
4. Add these variables:

| Variable | Secret? | Value |
| --- | --- | --- |
| `DATABRICKS_HOST` | No | Databricks workspace URL, for example `https://dbc-0f89edfa-de75.cloud.databricks.com` |
| `DATABRICKS_CLIENT_ID` | Usually yes | Service principal/client application ID |
| `STAGING_WORKSPACE_TOKEN` | Yes | OAuth client secret used for staging and test target runs |
| `PROD_WORKSPACE_TOKEN` | Yes | OAuth client secret used for prod deployment |

5. Save the variable group. If Azure DevOps asks for pipeline permissions, authorize the pipelines that use it.
6. Open `Pipelines > Environments` and create:
   - `mlops-staging`
   - `mlops-prod`
7. Optional but recommended: add an approval check to `mlops-prod`.
8. Optional but recommended: add an exclusive lock check to both environments to mirror the GitHub `concurrency` behavior.

## Create the Pipelines

Create three pipelines from existing YAML files:

1. `Pipelines > New pipeline`.
2. Select `Azure Repos Git`.
3. Select this repository.
4. Select `Existing Azure Pipelines YAML file`.
5. Choose one YAML file and save the pipeline:
   - `/azure-pipelines-ci.yml`
   - `/azure-pipelines-cd-staging.yml`
   - `/azure-pipelines-cd-prod.yml`
6. Rename the pipelines to something clear, for example:
   - `mlops_dbx-ci`
   - `mlops_dbx-cd-staging`
   - `mlops_dbx-cd-prod`

## Configure PR Validation

Azure Repos PR validation is configured as a branch policy, not only through YAML.

For `main`:

1. Go to `Repos > Branches`.
2. Open the `...` menu for `main`.
3. Select `Branch policies`.
4. Under `Build validation`, select `+`.
5. Select the `mlops_dbx-ci` pipeline.
6. Set trigger to `Automatic`.
7. Set policy requirement to `Required`.
8. Use this path filter:

```text
/mlops_dbx/*;/azure-pipelines-ci.yml
```

Repeat for `release` if pull requests are also used to promote changes into the release branch.

## Test the Migration

1. Manual CI smoke test:
   - Open `Pipelines > mlops_dbx-ci > Run pipeline`.
   - Select a branch that contains these YAML files.
   - Confirm all four jobs run: unit tests, integration test, staging validation, and prod validation.
2. PR validation test:
   - Create a branch.
   - Change a file under `mlops_dbx/`.
   - Open a pull request into `main`.
   - Confirm the branch policy queues `mlops_dbx-ci` and blocks completion until it passes.
3. Staging CD test:
   - Merge a passing PR into `main`, or manually run `mlops_dbx-cd-staging`.
   - Confirm the staging pipeline runs `databricks bundle validate -t staging` and `databricks bundle deploy -t staging`.
4. Prod CD test:
   - Merge or push to `release`, or manually run `mlops_dbx-cd-prod`.
   - Approve the `mlops-prod` environment if approval is configured.
   - Confirm the prod pipeline runs `databricks bundle validate -t prod` and `databricks bundle deploy -t prod`.

## Notes

- `azure-pipelines-ci.yml` validates the prod bundle with `STAGING_WORKSPACE_TOKEN` because the current GitHub Actions workflow does the same. The prod deployment pipeline uses `PROD_WORKSPACE_TOKEN`.
- The Databricks CLI is pinned to `0.236.0` to match the existing GitHub Actions setup.
- If a run fails before Databricks commands start, check variable group authorization and environment authorization first.
- If PR validation does not run in Azure Repos, recheck the branch build-validation policy. YAML `pr:` triggers are not used for Azure Repos Git.

## References

- Azure Pipelines variable groups: https://learn.microsoft.com/en-us/azure/devops/pipelines/library/variable-groups
- Azure Repos build validation policies: https://learn.microsoft.com/en-us/azure/devops/repos/git/branch-policies
- Azure Pipelines environments: https://learn.microsoft.com/en-us/azure/devops/pipelines/process/environments
- Azure Pipelines approvals and checks: https://learn.microsoft.com/en-us/azure/devops/pipelines/process/approvals
- Databricks CI/CD with Azure DevOps: https://docs.databricks.com/aws/en/dev-tools/ci-cd/azure-devops

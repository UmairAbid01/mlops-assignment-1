from mlflow.tracking import MlflowClient
import pandas as pd

results_df = pd.read_csv("results/model_results.csv")
# Find best model based on F1-score
best_model_name = results_df.loc[results_df["F1-Score"].idxmax(), "Model"]
print(f"\nBest Model: {best_model_name}")

client = MlflowClient()

# Get last run for the best model
runs = client.search_runs(
    experiment_ids=[client.get_experiment_by_name("Iris Classification").experiment_id],
    filter_string=f'tags.mlflow.runName = "{best_model_name}"',
    order_by=["start_time desc"],
    max_results=1
)
best_run_id = runs[0].info.run_id

# Register model in registry
model_uri = f"runs:/{best_run_id}/{best_model_name.replace(' ', '_')}"
registered_model = client.create_registered_model(best_model_name.replace(" ", "_"))
client.create_model_version(
    name=best_model_name.replace(" ", "_"),
    source=model_uri,
    run_id=best_run_id
)

print(f"\n✅ Registered {best_model_name} into MLflow Model Registry")

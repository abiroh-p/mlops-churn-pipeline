import os

project_name = "mlops-churn-pipeline"

folders = [
    "data",
    "dags",
    "src",
    "serving",
    "k8s",
    "k8s/airflow",
    "k8s/monitoring",
    "tests",
    ".github/workflows",
]

files = [
    "src/ingest.py",
    "src/validate.py",
    "src/features.py",
    "src/train.py",
    "src/evaluate.py",
    "src/register_model.py",
    "serving/app.py",
    "serving/Dockerfile",
    "k8s/mlflow-deployment.yaml",
    "k8s/serving-deployment.yaml",
    "dvc.yaml",
    "Dockerfile.train",
    "docker-compose.yaml",
    "README.md",
]

# Create root project folder
os.makedirs(project_name, exist_ok=True)

# Create folders
for folder in folders:
    path = os.path.join(project_name, folder)
    os.makedirs(path, exist_ok=True)

# Create files
for file in files:
    path = os.path.join(project_name, file)

    if not os.path.exists(path):
        with open(path, "w") as f:
            if path.endswith(".py"):
                f.write('"""Module placeholder."""\n')
            elif path.endswith(".yaml"):
                f.write("# YAML configuration\n")
            elif path.endswith(".md"):
                f.write(f"# {project_name}\n")
            elif path.endswith("Dockerfile"):
                f.write("# Docker configuration\n")
            else:
                f.write("")

print(f"✅ Project '{project_name}' created successfully!")
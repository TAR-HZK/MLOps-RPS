FROM astrocrpublic.azurecr.io/runtime:3.1-5

# Add MLflow
RUN pip install mlflow

# Expose MLflow port
EXPOSE 5000

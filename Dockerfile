FROM astrocrpublic.azurecr.io/runtime:3.1-5

# Switch to root for installing system packages
USER root

# Install git and clean up apt cache to avoid permission issues
RUN mkdir -p /var/lib/apt/lists/partial \
    && apt-get update \
    && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

# Switch back to default user (astro)
USER astro

# Install Python packages
RUN pip install mlflow dvc[s3]

# Expose MLflow port
EXPOSE 5000

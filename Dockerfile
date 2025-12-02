FROM astrocrpublic.azurecr.io/runtime:3.1-5

# 1. Switch to root to install OS-level tools
USER root

# Install git and clean up apt cache (keeps image small)
RUN mkdir -p /var/lib/apt/lists/partial \
    && apt-get update \
    && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

# CRITICAL: Tell Git to trust the Airflow directory
# (Required for the "Auto-Commit" feature in your DAG)
RUN git config --system --add safe.directory /usr/local/airflow

# 2. Switch back to the default Airflow user
USER astro

# 3. Install Python Packages
# We install heavy libraries here to avoid timeouts.
# - sweetviz: Required for the quality report
# - numpy<2.0: Required because Sweetviz crashes on Numpy 2.0+
RUN pip install --no-cache-dir --default-timeout=1000 \
    mlflow \
    dvc[s3] \
    sweetviz \
    "numpy<2.0"

# Expose MLflow port
EXPOSE 5000
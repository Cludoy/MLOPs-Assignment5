FROM python:3.10-slim

ARG RUN_ID=unknown
ENV RUN_ID=${RUN_ID}

WORKDIR /app

# Simulate downloading model artifacts from MLflow registry/tracking server.
CMD ["sh", "-c", "echo Downloading model for Run ID: ${RUN_ID}"]

FROM mambaorg/micromamba:1.5.8

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER
WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml .
RUN micromamba install -y -n base -f environment.yml && \
    micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

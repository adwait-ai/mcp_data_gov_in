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

# Expose port for HTTP server (will be set by Smithery via PORT env var)
EXPOSE 8000

# Run MCP server - it will auto-detect HTTP mode via PORT environment variable
CMD ["python", "mcp_server.py"]

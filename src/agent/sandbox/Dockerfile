# Execution sandbox for the agent's bash tool.
# Runs with --network none, so everything a task needs must be baked in here.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ripgrep \
    && apt-get clean && rm -r /var/lib/apt/lists/* \
    && pip install --no-cache-dir pytest==9.2.0

WORKDIR /workspace
CMD ["sleep", "infinity"]

FROM python:3.12-slim

# Install bash and common utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends bash && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory (required by DockerExecutionEnvironment)
WORKDIR /workspace

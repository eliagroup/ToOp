FROM python:3.11.4-bullseye

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Declare environment variables
ENV PATH="/home/$USERNAME/.local/bin:/root/.local/bin:$PATH"
ENV POETRY_VERSION="2.1.1"
ENV PROTOBUF_VERSION="33.1"

# Add user and usergroup
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -ms /bin/bash $USERNAME

# Install system packages
RUN apt-get -qq update && apt-get -qq -y install curl vim zip unzip htop \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

# Switch to non-root user before installing user tools
USER $USERNAME
WORKDIR /app

# Install user-level tools as vscode user
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && /home/$USERNAME/.local/bin/poetry config virtualenvs.create false \
    && PB_REL="https://github.com/protocolbuffers/protobuf/releases" \
    && curl -LO $PB_REL/download/v${PROTOBUF_VERSION}/protoc-${PROTOBUF_VERSION}-linux-x86_64.zip \
    && unzip protoc-${PROTOBUF_VERSION}-linux-x86_64.zip -d /home/$USERNAME/.local \
    && rm protoc-${PROTOBUF_VERSION}-linux-x86_64.zip


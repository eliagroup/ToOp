FROM python:3.11.4-bullseye

# Declare environment variables
ENV PATH="/root/.local/bin:$PATH"
ENV POETRY_VERSION="2.1.1"

# Install Poetry
RUN apt-get -qq update && apt-get -qq -y install curl vim zip htop\
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.create false \
    && apt-get -qq -y remove curl \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

# Install protoc compiler
ENV PROTOBUF_VERSION="33.1"
RUN PB_REL="https://github.com/protocolbuffers/protobuf/releases" \
    && curl -LO $PB_REL/download/v${PROTOBUF_VERSION}/protoc-${PROTOBUF_VERSION}-linux-x86_64.zip \
    && unzip protoc-${PROTOBUF_VERSION}-linux-x86_64.zip -d $HOME/.local \
    && rm protoc-${PROTOBUF_VERSION}-linux-x86_64.zip
WORKDIR /app

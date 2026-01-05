FROM python:3.11.4-bullseye@sha256:4b3c9c338fdf1db596eb1ccf83597b879098aecf30479a9f01839ab1f1cf0772                    

# Declare environment variables
ENV PATH="/root/.local/bin:$PATH"
ENV POETRY_VERSION="2.1.1"
ENV PROTOBUF_VERSION="33.1"

# Install tooling and protoc, then clean up build deps
RUN apt-get -qq update && apt-get -qq -y install curl vim zip unzip htop\
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.create false \
    && PB_REL="https://github.com/protocolbuffers/protobuf/releases" \
    && curl -LO $PB_REL/download/v${PROTOBUF_VERSION}/protoc-${PROTOBUF_VERSION}-linux-x86_64.zip \
    && unzip protoc-${PROTOBUF_VERSION}-linux-x86_64.zip -d $HOME/.local \
    && rm protoc-${PROTOBUF_VERSION}-linux-x86_64.zip \
    && apt-get -qq -y remove curl unzip \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

WORKDIR /app

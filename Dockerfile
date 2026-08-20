FROM ghcr.io/astral-sh/uv:0.11.19@sha256:b46b03ddfcfbf8f547af7e9eaefdf8a39c8cebcba7c98858d3162bd28cf536f6 AS uv

FROM python:3.13.6-bullseye@sha256:f58f33e0563f2ba81c7afe6259cd912f0c33413da93c75cc3a70a941c17afa8c

# Declare environment variables
ENV PATH="/root/.local/bin:$PATH"

COPY --from=uv /uv /usr/local/bin/uv

# Install tooling
RUN apt-get -qq update && apt-get -qq -y install vim zip unzip htop \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

WORKDIR /app

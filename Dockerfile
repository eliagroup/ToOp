FROM ghcr.io/astral-sh/uv:0.13.8@sha256:5cc1805e2426484abca4ad46ea783bab281fdb225fd82d8ce6d66a81254bac85 AS uv

FROM python:3.14.7@sha256:be8ccd085666c34273c9dc5607c9842f8b2e3116128aae45148ce164c07ce09d

# Declare environment variables
ENV PATH="/root/.local/bin:$PATH"

COPY --from=uv /uv /usr/local/bin/uv

# Install tooling
RUN apt-get -qq update && apt-get -qq -y install vim zip unzip htop \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

WORKDIR /app

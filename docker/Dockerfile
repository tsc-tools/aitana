ARG D_UID="1001"
ARG D_GID="1001"

FROM condaforge/miniforge3:25.11.0-1 AS base

ARG D_UID
ARG D_GID

USER root

COPY /docker/fix-permissions /usr/local/bin/fix-permissions

# Configure environment
ENV SHELL=/bin/bash \
    D_UID=$D_UID \
    D_GID=$D_GID \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    LANGUAGE=C.UTF-8 \
    HOME=/home/bayes

RUN groupadd -g $D_GID bayes && \
    useradd -m -s /bin/bash -N -u $D_UID -G bayes bayes && \
    chmod a+rx /usr/local/bin/fix-permissions && \
    mkdir $HOME/data && \
    fix-permissions $HOME && \
    fix-permissions $HOME/data && \
    fix-permissions /opt/conda && \
    echo "bayes:100000:65536" >> /etc/subuid && \
    echo "bayes:100000:65536" >> /etc/subgid

# Install Apptainer
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
        curl \
        squashfs-tools \
        fuse3 \
        fakeroot \
        tzdata \
        fonts-crosextra-carlito \
        uidmap && \
    curl -fsSL "https://github.com/apptainer/apptainer/releases/download/v1.4.5/apptainer_1.4.5_amd64.deb" \
        -o /tmp/apptainer.deb && \
    apt-get install --yes /tmp/apptainer.deb && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    rm /tmp/apptainer.deb


USER bayes

EXPOSE 8003

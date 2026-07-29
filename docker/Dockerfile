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

# Install Node.js (for npx) via conda-forge
RUN conda install -y -n base -c conda-forge nodejs=24.18.0 && \
    conda clean -afy && \
    fix-permissions /opt/conda

# Install uv system-wide
RUN curl -fsSL https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh

# Keep the virtual environment outside the bind-mounted workspace: it's
# platform-specific and this path lets a named volume mounted here inherit
# bayes ownership from the image, instead of the root ownership Docker gives
# a volume mounted inside the bind-mounted workspace.
ENV UV_PROJECT_ENVIRONMENT=$HOME/.venv

USER bayes
RUN mkdir -p $HOME/.venv && \
    echo 'eval "$(uv generate-shell-completion bash)"' >> ${HOME}/.bashrc

EXPOSE 8003
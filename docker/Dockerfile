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
    mkdir $HOME/.venv && \
    mkdir $HOME/.claude && \
    fix-permissions $HOME && \
    fix-permissions $HOME/data && \
    fix-permissions $HOME/.venv && \
    fix-permissions $HOME/.claude && \
    fix-permissions /opt/conda

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
        curl \
        tzdata \
        fonts-crosextra-carlito && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

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
# Put the workspace's console scripts (aitana-pipelines, aitana-bot, ...) on PATH
# in every stage, not only in shells VS Code happens to activate.
ENV PATH=$UV_PROJECT_ENVIRONMENT/bin:$PATH

USER bayes
RUN echo 'eval "$(uv generate-shell-completion bash)"' >> ${HOME}/.bashrc


EXPOSE 8003

# aitana-pipelines workflows run some rules in Apptainer containers
# (scripts/run_egu2026.sh builds with --target pipelines).
FROM base AS pipelines

USER root

# Install Apptainer
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
        squashfs-tools \
        fuse3 \
        fakeroot \
        uidmap && \
    curl -fsSL "https://github.com/apptainer/apptainer/releases/download/v1.4.5/apptainer_1.4.5_amd64.deb" \
        -o /tmp/apptainer.deb && \
    apt-get install --yes /tmp/apptainer.deb && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    rm /tmp/apptainer.deb && \
    echo "bayes:100000:65536" >> /etc/subuid && \
    echo "bayes:100000:65536" >> /etc/subgid

USER bayes

# Devcontainer image: the devcontainer bind-mounts the workspace and runs
# `uv sync` via postCreateCommand (devcontainer.json builds with --target dev).
FROM pipelines AS dev

USER root

# Quarto: renders .qmd reports to HTML, including executed Python code
# blocks (via the Jupyter engine). The .deb bundles its own Pandoc
# and Deno, so no extra system deps are required.
RUN curl -fsSL -o /tmp/quarto.deb \
      https://github.com/quarto-dev/quarto-cli/releases/download/v1.9.38/quarto-1.9.38-linux-amd64.deb && \
    apt-get update && apt-get install -y --no-install-recommends /tmp/quarto.deb openssh-client && \
    rm -f /tmp/quarto.deb && rm -rf /var/lib/apt/lists/*

USER bayes

# Operational image, and the default target (kept last so a plain
# `docker build`, as in .gitlab-ci.yml, produces it). Built from base, so
# BuildKit skips the bench and dev stages.
FROM base AS operational

# One image for every deployed service (storage API, pipelines, bot): all
# workspace packages, without the dev group or extras (so without torch).

# ~/.cache is root-owned (created by conda in base), and a uv cache would
# only bloat the image layers anyway.
ENV UV_NO_CACHE=1

# Install dependencies before copying the source so this layer stays cached
# until uv.lock changes; with --frozen, uv needs only these two files here.
COPY --chown=bayes:bayes uv.lock pyproject.toml /home/bayes/aitana/
WORKDIR /home/bayes/aitana
RUN uv sync --frozen --no-dev --all-packages --no-install-workspace

COPY --chown=bayes:bayes . /home/bayes/aitana
RUN uv sync --frozen --no-dev --all-packages

CMD ["tonik_api"]
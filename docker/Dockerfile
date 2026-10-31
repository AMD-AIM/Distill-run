ARG BASE_IMAGE=docker.io/rocm/pytorch@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG ENGINES=swift,trl,easydistill,yolo,ultralytics
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
# opencv-python (pulled in by ultralytics) needs libGL; setuptools>=77.0.3 is required by the pinned TRL
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" "setuptools>=77.0.3,<80" wheel
WORKDIR /opt/distill-run
COPY . /opt/distill-run
RUN PIP_NO_CACHE_DIR=1 python3 scripts/setup_envs.py --profile rocm714 --engines "${ENGINES}" \
      --core-dir .venvs/core --skip-submodules --index-url "${PIP_INDEX_URL}" \
    && for env in .venvs/*/; do name=$(basename "$env"); "$env/bin/python" -m pip freeze --all > ".distill/freeze-$name.txt"; done
ENV DISTILL_RUN_ENVS_ROOT=/opt/distill-run/.venvs \
    PATH=/opt/distill-run/.venvs/core/bin:${PATH} \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_DISABLE_XET=1 \
    PYTHONUNBUFFERED=1
# These change on every commit; declared before a RUN they would invalidate its cache.
ARG VCS_REF=unknown
ARG IMAGE_VERSION=dev
ARG PARAMS_SHA256=unknown
LABEL org.opencontainers.image.title="distill-run" \
      org.opencontainers.image.source="https://github.com/AMD-AIM/Distill-run" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      io.amd.rocm.arch="gfx1100" \
      distill-run.gpu-mode="single-gpu" \
      distill-run.params-sha256="${PARAMS_SHA256}"
ENTRYPOINT ["distill-run"]
CMD ["--help"]

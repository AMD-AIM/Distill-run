#!/usr/bin/env bash
# Build, tag, push and record the gfx1100 distill-run image from a clean HEAD.
#
#   scripts/build-image.sh [--no-push]
#
# Environment: IMAGE_REPO, BASE_IMAGE, ENGINES, PIP_INDEX_URL, CCR_USERNAME, CCR_PASSWORD.
# CCR_PASSWORD is optional (an existing `docker login` is reused) and only ever
# reaches docker through `docker login --password-stdin`.
#
# Exit codes: 0 ok; 1 uncommitted changes; 2 usage; 3 submodule sources missing;
# 4 pushed but no digest for IMAGE_REPO; anything else is docker's own exit code.
# The last lines on stdout are IMAGE=..., DIGEST=... (push only) and PARAMS_SHA256=...;
# docker's own output goes to stderr.
set -euo pipefail

IMAGE_REPO=${IMAGE_REPO:-ccr.ccs.tencentyun.com/radeon/distill-run}
BASE_IMAGE=${BASE_IMAGE:-docker.io/rocm/pytorch@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e}
ENGINES=${ENGINES:-swift,trl,easydistill,yolo,ultralytics}
PIP_INDEX_URL=${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}
CCR_USERNAME=${CCR_USERNAME:-100053684421}

push=1
for arg in "$@"; do
    case "$arg" in
        --no-push) push=0 ;;
        -h | --help)
            sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "usage: $0 [--no-push]" >&2
            exit 2
            ;;
    esac
done

cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain --ignore-submodules=none)" ]; then
    echo "build-image: uncommitted changes or untracked files; commit or remove them first:" >&2
    git status --short --ignore-submodules=none >&2
    exit 1
fi
if git submodule status | grep -q '^[-+U]'; then
    echo "build-image: submodules are not at their recorded commits:" >&2
    git submodule status >&2
    exit 3
fi
for name in trl easydistill; do
    if [ ! -f "third_party/$name/pyproject.toml" ]; then
        echo "build-image: third_party/$name is not checked out (git submodule update --init)" >&2
        exit 3
    fi
done

sha=$(git rev-parse HEAD)
tag="$(git rev-parse --short=12 HEAD)-gfx1100"
params=$(sha256sum src/distill_run/params.py | cut -d' ' -f1)
image="$IMAGE_REPO:$tag"

docker build -f docker/Dockerfile \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    --build-arg "ENGINES=$ENGINES" \
    --build-arg "PIP_INDEX_URL=$PIP_INDEX_URL" \
    --build-arg "VCS_REF=$sha" \
    --build-arg "IMAGE_VERSION=$tag" \
    --build-arg "PARAMS_SHA256=$params" \
    -t "$image" . >&2

digest=
if [ "$push" = 1 ]; then
    if [ -n "${CCR_PASSWORD:-}" ]; then
        printf '%s' "$CCR_PASSWORD" |
            docker login "${IMAGE_REPO%%/*}" --username "$CCR_USERNAME" --password-stdin >&2
    fi
    docker push "$image" >&2
    digest=$(docker inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$image" |
        grep -m1 "^$IMAGE_REPO@sha256:" || true)
    if [ -z "$digest" ]; then
        echo "build-image: pushed $image but docker reports no $IMAGE_REPO digest" >&2
        exit 4
    fi
    mkdir -p build
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$tag" "$sha" "$digest" \
        "$BASE_IMAGE" >> build/image-digests.tsv
fi

echo "IMAGE=$image"
if [ -n "$digest" ]; then
    echo "DIGEST=$digest"
fi
echo "PARAMS_SHA256=$params"

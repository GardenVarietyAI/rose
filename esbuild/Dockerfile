FROM alpine:3.20

ARG ESBUILD_VERSION=0.25.0
ARG TARGETARCH

RUN apk add --no-cache ca-certificates curl tar

RUN set -eu; \
  case "${TARGETARCH:-}" in \
    amd64) pkg_arch="linux-x64" ;; \
    arm64) pkg_arch="linux-arm64" ;; \
    *) echo "unsupported TARGETARCH: ${TARGETARCH:-<unset>}" >&2; exit 1 ;; \
  esac; \
  url="https://registry.npmjs.org/@esbuild/${pkg_arch}/-/${pkg_arch}-${ESBUILD_VERSION}.tgz"; \
  curl -fsSL "$url" -o /tmp/esbuild.tgz; \
  mkdir -p /tmp/esbuild && tar -xzf /tmp/esbuild.tgz -C /tmp/esbuild; \
  mkdir -p /usr/local/bin; \
  cp /tmp/esbuild/package/bin/esbuild /usr/local/bin/esbuild; \
  chmod 0755 /usr/local/bin/esbuild; \
  rm -rf /tmp/esbuild /tmp/esbuild.tgz

WORKDIR /app
ENTRYPOINT ["esbuild"]

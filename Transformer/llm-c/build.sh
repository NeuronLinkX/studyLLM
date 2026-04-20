#!/usr/bin/env bash
# llm-c/build.sh
set -euo pipefail

# ======== 콘솔 색상 ========
c_green(){ printf "\033[32m%s\033[0m\n" "$*"; }
c_yell(){  printf "\033[33m%s\033[0m\n" "$*"; }
c_red(){   printf "\033[31m%s\033[0m\n" "$*"; }
c_dim(){   printf "\033[2m%s\033[0m\n" "$*"; }

# ======== 기본 설정 ========
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
CMAKE_BUILD_TYPE="Release"
DO_CLEAN=0
DO_RUN=1
DO_WEIGHTS=1
VERBOSE=0
FORCE_ARCH=""

usage() {
  cat <<'EOF'
Usage: ./build.sh [options]
  --clean          : build/ 삭제 후 재생성
  --no-run         : 실행 생략
  --no-weights     : weights.bin 생성 생략
  --debug          : Debug 빌드
  --arch {arm64|x86_64}
  --verbose
  -h, --help
EOF
}

# ======== 인자 파싱 ========
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) DO_CLEAN=1 ;;
    --no-run) DO_RUN=0 ;;
    --no-weights) DO_WEIGHTS=0 ;;
    --debug) CMAKE_BUILD_TYPE="Debug" ;;
    --arch) shift; FORCE_ARCH="${1:-}";;
    --verbose) VERBOSE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) c_red "Unknown option: $1"; usage; exit 1 ;;
  esac
  shift
done

host_uname="$(uname -s || true)"
host_arch="$(uname -m || true)"

detect_arch() {
  if [[ -n "$FORCE_ARCH" ]]; then echo "$FORCE_ARCH"; return; fi
  if [[ "$host_uname" == "Darwin" ]]; then
    [[ "$host_arch" == "arm64" ]] && echo "arm64" || echo "x86_64"
  else
    echo "$host_arch"
  fi
}
TARGET_ARCH="$(detect_arch)"
c_dim "Host: ${host_uname} ${host_arch}  |  Target: ${TARGET_ARCH}"

# ======== 클린 ========
if [[ $DO_CLEAN -eq 1 ]]; then
  c_yell "[clean] removing ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

# ======== CMake 인자 구성 ========
CMAKE_ARGS=(-S "${PROJECT_ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

if [[ "$host_uname" == "Darwin" ]]; then
  if [[ "$TARGET_ARCH" == "arm64" ]]; then
    CMAKE_ARGS+=(-DCMAKE_OSX_ARCHITECTURES=arm64)
  else
    CMAKE_ARGS+=(-DCMAKE_OSX_ARCHITECTURES=x86_64)
  fi
fi

# 배열 대신 문자열 플래그 사용 (set -u 안전)
BUILD_VERBOSE=""
if [[ $VERBOSE -eq 1 ]]; then
  BUILD_VERBOSE="--verbose"
fi

run_cmake_config() {
  if [[ "$host_uname" == "Darwin" && "$TARGET_ARCH" == "arm64" && "$host_arch" != "arm64" ]]; then
    arch -arm64 bash -lc "$(printf '%q ' cmake "${CMAKE_ARGS[@]}")"
  else
    cmake "${CMAKE_ARGS[@]}"
  fi
}

run_cmake_build() {
  if [[ "$host_uname" == "Darwin" && "$TARGET_ARCH" == "arm64" && "$host_arch" != "arm64" ]]; then
    arch -arm64 bash -lc "cmake --build $(printf '%q ' "${BUILD_DIR}") -j ${BUILD_VERBOSE}"
  else
    cmake --build "${BUILD_DIR}" -j ${BUILD_VERBOSE}
  fi
}

make_weights() {
  local OUT="${BUILD_DIR}/weights.bin"
  if [[ $DO_WEIGHTS -eq 0 ]]; then c_dim "[weights] skip"; return; fi
  local PY=python3
  if [[ -x "${PROJECT_ROOT}/.venv/bin/python3" ]]; then PY="${PROJECT_ROOT}/.venv/bin/python3"; fi
  c_yell "[weights] generating ${OUT}"
  "${PY}" "${PROJECT_ROOT}/scripts/gen_weights.py" \
    --d_model 6 --n_head 2 --d_ff 24 --seed 7 --out "${OUT}"
}

run_binaries() {
  if [[ $DO_RUN -eq 0 ]]; then c_dim "[run] skip"; return; fi
  local DEMO="${BUILD_DIR}/demo"
  local HELLO="${BUILD_DIR}/hello_min"
  if [[ -x "$DEMO" ]]; then c_green "[run] ${DEMO}"; "$DEMO"; else c_red "[run] demo not found"; fi
  if [[ -x "$HELLO" ]]; then c_green "[run] ${HELLO}"; "$HELLO"; else c_dim "[run] hello_min not found (optional)"; fi
}

c_green "[1/3] CMake configure → ${CMAKE_BUILD_TYPE}"
run_cmake_config

c_green "[2/3] Build"
run_cmake_build

c_green "[3/3] Weights & Run"
make_weights
run_binaries

c_green "Done."

#!/bin/bash
set -euo pipefail # Fail fast

BASE_DIR=$(dirname "$0")/..
SRC_DIR="$BASE_DIR/src"
BIN_DIR="$BASE_DIR/bin"

CFLAGS_SEQ="-O3 -march=native"
CFLAGS_OMP="-O3 -march=native -fopenmp"
CFLAGS_MPI="-O3 -march=native"

mkdir -p "$BIN_DIR"

compile_seq() {
    echo "Compiling Sequential Kernel using Makefile..."
    make -C "$SRC_DIR/seq" all
}

compile_openmp() {
    echo "Compiling OpenMP Kernel using Makefile..."
    make -C "$SRC_DIR/openmp" all
}

compile_mpi() {
    echo "Compiling MPI Kernel using Makefile..."
    make -C "$SRC_DIR/mpi" all
}

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <seq|openmp|mpi|all>"
    exit 1
fi

case "$1" in
    seq)
        compile_seq
        ;;
    openmp)
        compile_openmp
        ;;
    mpi)
        compile_mpi
        ;;
    all)
        compile_seq
        compile_openmp
        compile_mpi
        ;;
    *)
        echo "Invalid argument. Usage: $0 <seq|openmp|mpi|all>"
        exit 1
        ;;
esac

echo "Compilation finished."
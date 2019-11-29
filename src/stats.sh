#!/bin/bash

cargo build --release;
perf stat -B -e cache-references,cache-misses,cycles,instructions,branches,branch-misses,faults,migrations,context-switches ./target/release/gemm
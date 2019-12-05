#!/bin/bash
# export RUSTFLAGS="-C target-feature=-mmx,-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-3dnow,-3dnowa,-avx,-avx2" 
export RUSTFLAGS="-C target-feature=-mmx,-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-3dnow,-3dnowa,+avx,+avx2" 
#cargo clean && cargo build --release
perf stat -B -e cache-references,cache-misses,cycles,instructions,branches,branch-misses,faults,migrations,context-switches ./target/release/gemm

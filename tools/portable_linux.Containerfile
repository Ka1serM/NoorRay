#! glibc 2.28 is the oldest glibc that Blender supports; the module built here loads on it.
FROM quay.io/pypa/manylinux_2_28_x86_64
RUN dnf install -y clang lld llvm compiler-rt && dnf clean all && pipx install ninja
#! AlmaLinux 8 has no libc++, so the runtimes of the same LLVM release are built against its glibc.
ARG LLVM_VERSION=22.1.8
RUN curl -sSfL https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/llvm-project-${LLVM_VERSION}.src.tar.xz \
        | tar xJ -C /tmp \
    && cmake -S /tmp/llvm-project-${LLVM_VERSION}.src/runtimes -B /tmp/runtimes -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DLLVM_ENABLE_RUNTIMES="libunwind;libcxxabi;libcxx" \
        -DLIBUNWIND_ENABLE_SHARED=OFF -DLIBCXXABI_ENABLE_SHARED=OFF -DLIBCXX_ENABLE_SHARED=OFF \
        -DLIBCXXABI_USE_LLVM_UNWINDER=ON -DLIBCXXABI_USE_COMPILER_RT=ON -DLIBCXX_USE_COMPILER_RT=ON \
        -DLIBUNWIND_USE_COMPILER_RT=ON -DLIBCXX_INCLUDE_BENCHMARKS=OFF -DLIBCXX_INCLUDE_TESTS=OFF \
        -DLIBCXXABI_INCLUDE_TESTS=OFF -DLIBUNWIND_INCLUDE_TESTS=OFF \
    && ninja -C /tmp/runtimes install-unwind install-cxxabi install-cxx \
    && rm -rf /tmp/runtimes /tmp/llvm-project-${LLVM_VERSION}.src

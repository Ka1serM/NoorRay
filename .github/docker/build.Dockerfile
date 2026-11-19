# The image of the Linux, Windows and web builds (.github/workflows/build.yml). The workflow builds it again only when
# this file changes, so the jobs install no packages. Ubuntu 22.04 has glibc 2.35, so the Linux executable runs on older
# distributions too. Clang, LLD and libc++ come from LLVM's own packages for the release; llvm-mingw cross-compiles the
# Windows executable.
FROM ubuntu:22.04
LABEL org.opencontainers.image.source=https://github.com/Ka1serM/NoorRay

# actions/checkout needs git, get-cmake needs unzip, emsdk needs xz-utils, and the deploy needs brotli, rsync and ssh.
# SDL in RmlKit builds against Wayland, X11 through xkbcommon and EGL, and plays audio through PulseAudio.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git make diffutils python3 xz-utils zstd ccache curl ca-certificates gnupg \
        pkg-config libwayland-dev libxkbcommon-dev libegl-dev libpulse-dev \
        brotli rsync openssh-client unzip zip \
    && rm -rf /var/lib/apt/lists/*

# LLVM 22: Clang, LLD, libc++, compiler-rt and libunwind. clang-22 pulls in GCC 11's runtime, which the build does not use.
ARG LLVM_VERSION=22
RUN curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-${LLVM_VERSION} main" \
        > /etc/apt/sources.list.d/llvm.list \
    && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        clang-${LLVM_VERSION} lld-${LLVM_VERSION} llvm-${LLVM_VERSION} libc++-${LLVM_VERSION}-dev \
        libc++abi-${LLVM_VERSION}-dev libclang-rt-${LLVM_VERSION}-dev libunwind-${LLVM_VERSION}-dev \
    && rm -rf /var/lib/apt/lists/*

# llvm-mingw: Clang, LLD and libc++ for Windows.
ARG LLVM_MINGW_VERSION=20261020
RUN curl -fsSL "https://github.com/mstorsjo/llvm-mingw/releases/download/${LLVM_MINGW_VERSION}/llvm-mingw-${LLVM_MINGW_VERSION}-ucrt-ubuntu-22.04-x86_64.tar.xz" \
        | tar -xJ -C /opt && mv /opt/llvm-mingw-* /opt/llvm-mingw

# The cross builds compile their host tools with cc, such as the web and Windows builds' flatc.
ENV PATH=/usr/lib/llvm-${LLVM_VERSION}/bin:/opt/llvm-mingw/bin:$PATH
RUN ln -s /usr/lib/llvm-${LLVM_VERSION}/bin/clang /usr/bin/cc && ln -s /usr/lib/llvm-${LLVM_VERSION}/bin/clang++ /usr/bin/c++

# The workspace belongs to another user than the jobs, and git refuses such a repository by default.
RUN git config --system --add safe.directory '*'

ENV CC=clang CXX=clang++

cmake_minimum_required(VERSION 3.16)
project(jhq_repro CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# FAISS (pre-built from JHQ official repo)
# Override via -DJHQ_OFFICIAL_ROOT=<path> or set env JHQ_OFFICIAL_ROOT
if(DEFINED ENV{JHQ_OFFICIAL_ROOT})
    set(JHQ_OFFICIAL_ROOT $ENV{JHQ_OFFICIAL_ROOT})
elseif(NOT DEFINED JHQ_OFFICIAL_ROOT)
    if(APPLE)
        set(JHQ_OFFICIAL_ROOT /Users/apple/github/JHQ_official/jhq)
    else()
        set(JHQ_OFFICIAL_ROOT /root/JHQ_official/jhq)
    endif()
endif()

if(APPLE)
    set(_FAISS_BUILD_DIR ${JHQ_OFFICIAL_ROOT}/build_mac)
else()
    set(_FAISS_BUILD_DIR ${JHQ_OFFICIAL_ROOT}/build_linux)
endif()

set(FAISS_INCLUDE_DIRS ${JHQ_OFFICIAL_ROOT}/external/faiss)
set(FAISS_LIBRARIES    ${_FAISS_BUILD_DIR}/external/faiss/faiss/libfaiss.a)

message(STATUS "JHQ_OFFICIAL_ROOT: ${JHQ_OFFICIAL_ROOT}")
message(STATUS "FAISS_LIBRARIES:   ${FAISS_LIBRARIES}")

if(APPLE)
    # macOS: use Homebrew libomp
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY /usr/local/opt/libomp/lib/libomp.dylib)
endif()
find_package(OpenMP REQUIRED)

add_subdirectory(src)
add_subdirectory(examples)

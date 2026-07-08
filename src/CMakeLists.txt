add_library(jhq_impl STATIC
    jl_transform.cpp
    codebook.cpp
    jq_index.cpp
    jhq_index.cpp
    ivf.cpp
    jq_ivf_index.cpp
    jhq_ivf_index.cpp
)

target_include_directories(jhq_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FAISS_INCLUDE_DIRS}
)

target_link_libraries(jhq_impl PUBLIC
    ${FAISS_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    OpenMP::OpenMP_CXX
)

target_compile_options(jhq_impl PRIVATE -O3 -march=native -Wno-deprecated-declarations)

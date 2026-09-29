cmake_minimum_required(VERSION 3.15)
project(napypi LANGUAGES CXX)

# -------------------------
# Dependencies
# -------------------------
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# -------------------------
# Python extension
# -------------------------
pybind11_add_module(_core
    src/napy.cpp
    src/anova.cpp
    src/chi2.cpp
    src/kruskal.cpp
    src/matrix.cpp
    src/mwu.cpp
    src/pearson.cpp
    src/spearman.cpp
    src/ttest.cpp
    src/partialcorr.cpp
    src/multinomial_reg.cpp
    src/linear_reg.cpp
)

target_compile_features(_core PRIVATE cxx_std_17)

# Project headers
target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Eigen (vendored, header-only)
target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen3
)

# Boost (vendored, header-only)
target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/boost
)

# mlpack and its dependencies (vendored, header-only)
target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mlpack/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/armadillo
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ensmallen
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cereal
)

# OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
else()
    message(FATAL_ERROR "OpenMP required but not found!")
endif()

# Link BLAS and LAPACK libraries
target_link_libraries(_core PRIVATE 
    ${BLAS_LIBRARIES} 
    ${LAPACK_LIBRARIES}
)

# Install location inside Python package
install(TARGETS _core
        LIBRARY DESTINATION napypi
        RUNTIME DESTINATION napypi)

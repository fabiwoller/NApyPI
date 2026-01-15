cmake_minimum_required(VERSION 3.15)
project(napypi LANGUAGES CXX)

# -------------------------
# Dependencies
# -------------------------
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

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
)

target_compile_features(_core PRIVATE cxx_std_17)

# Project headers
target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Eigen (vendored, header-only)
target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen
)

# Boost (vendored, header-only)
target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# Install location inside Python package
install(TARGETS _core
        LIBRARY DESTINATION napypi
        RUNTIME DESTINATION napypi)

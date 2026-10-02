#pragma once

#include "matrix.hpp"
#include <utility>
#include <vector>

// Boundary between the OpenMP-parallel driver in multinomial_reg.cpp and the
// mlpack-backed worker in multinomial_reg_impl.cpp.
//
// mlpack's base.hpp refuses to compile whenever _OPENMP is defined but reports
// less than OpenMP 3.1, and MSVC only implements 2.0. Confining every mlpack
// include to its own translation unit lets CMake build that one file with
// /openmp- on MSVC while this driver keeps its #pragma omp parallel for, so
// multinomial and logistic regression stay multi-threaded on Windows.

enum class PredictorSource
{
    None,
    Categorical,
    Continuous
};

// Stores number of samples without NAs in all involved variables in num_samples.
std::pair<double, double> pairwise_nan_multinomial_regression(
    const DataMatrix& cat_data,
    const DataMatrix& cont_data,
    int dependent_idx,
    int independent_idx,
    const std::vector<int>& control_rows_categorical,
    const std::vector<int>& control_rows_continuous,
    double na_value,
    PredictorSource predictor_source,
    double& num_samples);

#include "stats.hpp"
#include "multinomial_reg_impl.hpp"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

using namespace std;

std::pair<DataMatrix, DataMatrix> statistics::multinomial_regression_test_with_nans(
    const DataMatrix& cat_data,
    const DataMatrix& cont_data,
    const std::vector<int>& control_rows_categorical,
    const std::vector<int>& control_rows_continuous,
    double na_value)
{
    const size_t num_cat_variables = cat_data.rows();
    const size_t num_cont_variables = cont_data.rows();
    const size_t num_comb_variables = num_cat_variables + num_cont_variables;
    DataMatrix correlations(num_cat_variables, num_comb_variables);
    DataMatrix pvalues(num_cat_variables, num_comb_variables);

    auto is_control_row = [](const std::vector<int>& control_rows, size_t row_index) {
        return std::find(control_rows.begin(), control_rows.end(), static_cast<int>(row_index)) != control_rows.end();
    };

    // MSVC only implements OpenMP 2.0, which requires a signed loop index, so the parallel
    // loop runs over a signed counter and converts back to size_t for the body.
    const std::ptrdiff_t num_cat_variables_signed = static_cast<std::ptrdiff_t>(num_cat_variables);
    #pragma omp parallel for
    for (std::ptrdiff_t dependent_row = 0; dependent_row < num_cat_variables_signed; ++dependent_row)
    {
        const size_t dependent_idx = static_cast<size_t>(dependent_row);
        if (is_control_row(control_rows_categorical, dependent_idx))
        {
            for (size_t independent_idx = 0; independent_idx < num_comb_variables; ++independent_idx)
            {
                correlations(dependent_idx, independent_idx) = std::numeric_limits<double>::quiet_NaN();
                pvalues(dependent_idx, independent_idx) = std::numeric_limits<double>::quiet_NaN();
            }
            continue;
        }

        for (size_t independent_idx = 0; independent_idx < num_cat_variables; ++independent_idx)
        {
            // Regressing a variable on itself is not a meaningful test.
            if (independent_idx == dependent_idx || is_control_row(control_rows_categorical, independent_idx))
            {
                correlations(dependent_idx, independent_idx) = std::numeric_limits<double>::quiet_NaN();
                pvalues(dependent_idx, independent_idx) = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

            const std::pair<double, double> results = pairwise_nan_multinomial_regression(
                cat_data,
                cont_data,
                dependent_idx,
                independent_idx,
                control_rows_categorical,
                control_rows_continuous,
                na_value,
                PredictorSource::Categorical);
            correlations(dependent_idx, independent_idx) = std::get<0>(results);
            pvalues(dependent_idx, independent_idx) = std::get<1>(results);
        }

        for (size_t independent_idx = 0; independent_idx < num_cont_variables; ++independent_idx)
        {
            if (is_control_row(control_rows_continuous, independent_idx))
            {
                correlations(dependent_idx, num_cat_variables + independent_idx) = std::numeric_limits<double>::quiet_NaN();
                pvalues(dependent_idx, num_cat_variables + independent_idx) = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

            const std::pair<double, double> results = pairwise_nan_multinomial_regression(
                cat_data,
                cont_data,
                dependent_idx,
                independent_idx,
                control_rows_categorical,
                control_rows_continuous,
                na_value,
                PredictorSource::Continuous);
            correlations(dependent_idx, num_cat_variables + independent_idx) = std::get<0>(results);
            pvalues(dependent_idx, num_cat_variables + independent_idx) = std::get<1>(results);
        }
    }

    return std::make_pair(correlations, pvalues);
}

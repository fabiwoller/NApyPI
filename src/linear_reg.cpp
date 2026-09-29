#include "stats.hpp"
#include <algorithm>
#include <eigen3/Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace
{

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

constexpr double RANK_TOLERANCE = 1e-7;

enum class PredictorSource
{
    Categorical,
    Continuous
};

struct OlsFit
{
    double rss = NaN;
    Eigen::Index rank = 0;
    VectorXd coefficients;
};

struct RegressionResult
{
    double f_statistic = NaN;
    double p_value = NaN;
    double np2 = NaN;
    double cohens_f2 = NaN;
    double beta = NaN;
    double std_beta = NaN;
};

std::vector<int> collect_valid_cols(
    const DataMatrix& cat_data,
    const DataMatrix& cont_data,
    int dependent_idx,
    int independent_idx,
    const std::vector<int>& control_rows_categorical,
    const std::vector<int>& control_rows_continuous,
    double na_value,
    PredictorSource predictor_source)
{
    std::vector<int> valid_cols;
    const size_t ncols = cont_data.cols();

    for (size_t c = 0; c < ncols; ++c) {
        bool valid = (cont_data(dependent_idx, c) != na_value);

        if (predictor_source == PredictorSource::Categorical) {
            valid &= (cat_data(independent_idx, c) != na_value);
        } else {
            valid &= (cont_data(independent_idx, c) != na_value);
        }

        for (auto z : control_rows_categorical) {
            valid &= (cat_data(z, c) != na_value);
        }
        for (auto z : control_rows_continuous) {
            valid &= (cont_data(z, c) != na_value);
        }

        if (valid) {
            valid_cols.push_back(static_cast<int>(c));
        }
    }

    return valid_cols;
}

// Returns the sorted distinct values of the given categorical row on the valid columns.
std::vector<double> collect_levels(const DataMatrix& data, int row, const std::vector<int>& valid_cols)
{
    std::vector<double> levels;
    levels.reserve(valid_cols.size());
    for (int col : valid_cols) {
        levels.push_back(data(row, col));
    }

    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    return levels;
}

// Adds one dummy column per level of the categorical row, using the lowest level as reference category.
void add_dummy_columns(
    const DataMatrix& cat_data,
    int row,
    const std::vector<double>& levels,
    const std::vector<int>& valid_cols,
    MatrixXd& design,
    Eigen::Index& column)
{
    for (size_t level_idx = 1; level_idx < levels.size(); ++level_idx, ++column) {
        for (size_t s = 0; s < valid_cols.size(); ++s) {
            design(s, column) = (cat_data(row, valid_cols[s]) == levels[level_idx]) ? 1.0 : 0.0;
        }
    }
}

// Builds the design matrix (observations as rows) of the full model. Columns are ordered as intercept,
// categorical confounders (dummy-encoded), continuous confounders, and the predictor columns last, such that
// the design matrix of the reduced model is given by the leading columns. Returns the number of predictor columns.
Eigen::Index build_design_matrix(
    const DataMatrix& cat_data,
    const DataMatrix& cont_data,
    int independent_idx,
    const std::vector<int>& control_rows_categorical,
    const std::vector<int>& control_rows_continuous,
    PredictorSource predictor_source,
    const std::vector<int>& valid_cols,
    MatrixXd& design)
{
    const Eigen::Index num_samples = valid_cols.size();
    Eigen::Index num_columns = 1 + control_rows_continuous.size();

    std::vector<std::vector<double>> confounder_levels;
    confounder_levels.reserve(control_rows_categorical.size());
    for (int row : control_rows_categorical) {
        confounder_levels.push_back(collect_levels(cat_data, row, valid_cols));
        num_columns += confounder_levels.back().size() - 1;
    }

    std::vector<double> predictor_levels;
    Eigen::Index predictor_columns = 1;
    if (predictor_source == PredictorSource::Categorical) {
        predictor_levels = collect_levels(cat_data, independent_idx, valid_cols);
        predictor_columns = predictor_levels.size() - 1;
    }
    num_columns += predictor_columns;

    design.resize(num_samples, num_columns);
    design.col(0).setOnes();
    Eigen::Index column = 1;

    for (size_t i = 0; i < control_rows_categorical.size(); ++i) {
        add_dummy_columns(cat_data, control_rows_categorical[i], confounder_levels[i], valid_cols, design, column);
    }

    for (int row : control_rows_continuous) {
        for (Eigen::Index s = 0; s < num_samples; ++s) {
            design(s, column) = cont_data(row, valid_cols[s]);
        }
        ++column;
    }

    if (predictor_source == PredictorSource::Categorical) {
        add_dummy_columns(cat_data, independent_idx, predictor_levels, valid_cols, design, column);
    } else {
        for (Eigen::Index s = 0; s < num_samples; ++s) {
            design(s, column) = cont_data(independent_idx, valid_cols[s]);
        }
    }

    return predictor_columns;
}

// Ordinary least squares via column-pivoted QR decomposition. Linearly dependent columns are detected and
// get a coefficient of zero, such that the rank reflects the actual degrees of freedom of the model.
OlsFit fit_ols(const MatrixXd& design, const VectorXd& y)
{
    // Scale columns to unit norm so that rank detection does not depend on the scale of the variables.
    VectorXd scale = design.colwise().norm().transpose();
    for (Eigen::Index j = 0; j < scale.size(); ++j) {
        if (scale(j) == 0.0) {
            scale(j) = 1.0;
        }
    }
    const MatrixXd scaled_design = design * scale.cwiseInverse().asDiagonal();

    Eigen::ColPivHouseholderQR<MatrixXd> qr(scaled_design.rows(), scaled_design.cols());
    qr.setThreshold(RANK_TOLERANCE);
    qr.compute(scaled_design);
    const Eigen::Index rank = qr.rank();

    // Solve R z = Q^T y on the leading rank pivoted columns only.
    const VectorXd qty = qr.householderQ().adjoint() * y;
    VectorXd z = VectorXd::Zero(design.cols());
    z.head(rank) = qr.matrixQR().topLeftCorner(rank, rank).triangularView<Eigen::Upper>().solve(qty.head(rank));

    OlsFit fit;
    fit.rank = rank;
    fit.coefficients = (qr.colsPermutation() * z).cwiseQuotient(scale);
    fit.rss = (y - design * fit.coefficients).squaredNorm();
    return fit;
}

double sum_of_squared_deviations(const VectorXd& x)
{
    return (x.array() - x.mean()).square().sum();
}

// Compares the full model y ~ 1 + confounders + predictor against the reduced model y ~ 1 + confounders
// on all samples without missing values in any of the involved variables.
RegressionResult pairwise_nan_linear_regression(
    const DataMatrix& cat_data,
    const DataMatrix& cont_data,
    int dependent_idx,
    int independent_idx,
    const std::vector<int>& control_rows_categorical,
    const std::vector<int>& control_rows_continuous,
    double na_value,
    PredictorSource predictor_source)
{
    RegressionResult result;

    const std::vector<int> valid_cols = collect_valid_cols(
        cat_data,
        cont_data,
        dependent_idx,
        independent_idx,
        control_rows_categorical,
        control_rows_continuous,
        na_value,
        predictor_source);

    const Eigen::Index num_samples = valid_cols.size();
    if (num_samples == 0) {
        return result;
    }

    VectorXd y(num_samples);
    for (Eigen::Index s = 0; s < num_samples; ++s) {
        y(s) = cont_data(dependent_idx, valid_cols[s]);
    }

    // Effect sizes are undefined for a constant dependent variable.
    if ((y.array() == y(0)).all()) {
        return result;
    }

    MatrixXd design;
    const Eigen::Index predictor_columns = build_design_matrix(
        cat_data,
        cont_data,
        independent_idx,
        control_rows_categorical,
        control_rows_continuous,
        predictor_source,
        valid_cols,
        design);

    // Categorical predictor with a single remaining category.
    if (predictor_columns == 0) {
        return result;
    }

    const OlsFit full = fit_ols(design, y);
    const OlsFit reduced = fit_ols(design.leftCols(design.cols() - predictor_columns), y);

    const Eigen::Index df_effect = full.rank - reduced.rank;
    const Eigen::Index df_residual = num_samples - full.rank;
    const double total_ss = sum_of_squared_deviations(y);

    // Predictor is collinear with the confounders, too few samples, or the dependent variable is already
    // perfectly explained by the confounders.
    if (df_effect <= 0 || df_residual <= 0 || reduced.rss <= std::numeric_limits<double>::epsilon() * total_ss) {
        return result;
    }

    const double effect_ss = std::max(reduced.rss - full.rss, 0.0);
    result.np2 = effect_ss / reduced.rss;

    if (full.rss > 0.0) {
        result.cohens_f2 = effect_ss / full.rss;
        result.f_statistic = (effect_ss / df_effect) / (full.rss / df_residual);
    } else {
        result.cohens_f2 = std::numeric_limits<double>::infinity();
        result.f_statistic = std::numeric_limits<double>::infinity();
    }

    // Boost throws on non-finite arguments, which must not happen inside an OpenMP region.
    if (std::isfinite(result.f_statistic)) {
        boost::math::fisher_f dist(static_cast<double>(df_effect), static_cast<double>(df_residual));
        result.p_value = boost::math::cdf(boost::math::complement(dist, result.f_statistic));
    } else {
        result.p_value = 0.0;
    }

    // A single slope only exists for continuous or binary predictors.
    if (predictor_columns == 1) {
        const VectorXd x = design.col(design.cols() - 1);
        result.beta = full.coefficients(design.cols() - 1);
        result.std_beta = result.beta * std::sqrt(sum_of_squared_deviations(x) / total_ss);
    }

    return result;
}

bool is_control_row(const std::vector<int>& control_rows, size_t row_index)
{
    return std::find(control_rows.begin(), control_rows.end(), static_cast<int>(row_index)) != control_rows.end();
}

DataMatrix* find_matrix(std::map<std::string, DataMatrix>& output, const std::string& name)
{
    auto it = output.find(name);
    return it == output.end() ? nullptr : &it->second;
}

} // namespace

std::map<std::string, DataMatrix> statistics::linear_regression_with_nans(
    const DataMatrix& cat_data,
    const DataMatrix& cont_data,
    const std::vector<int>& control_rows_categorical,
    const std::vector<int>& control_rows_continuous,
    double na_value,
    const std::set<std::string>& return_types)
{
    const size_t num_cat_variables = cat_data.rows();
    const size_t num_cont_variables = cont_data.rows();
    const size_t num_comb_variables = num_cat_variables + num_cont_variables;

    // Only create the requested output matrices.
    std::map<std::string, DataMatrix> output;
    for (const std::string name : {"F", "p_unadjusted", "np2", "cohens_f2", "beta", "std_beta"}) {
        if (return_types.count(name)) {
            output.emplace(name, DataMatrix(num_cont_variables, num_comb_variables));
        }
    }
    DataMatrix* f_stat = find_matrix(output, "F");
    DataMatrix* pvalues = find_matrix(output, "p_unadjusted");
    DataMatrix* np2 = find_matrix(output, "np2");
    DataMatrix* cohens_f2 = find_matrix(output, "cohens_f2");
    DataMatrix* beta = find_matrix(output, "beta");
    DataMatrix* std_beta = find_matrix(output, "std_beta");

    auto store = [&](size_t row, size_t col, const RegressionResult& result) {
        if (f_stat) (*f_stat)(row, col) = result.f_statistic;
        if (pvalues) (*pvalues)(row, col) = result.p_value;
        if (np2) (*np2)(row, col) = result.np2;
        if (cohens_f2) (*cohens_f2)(row, col) = result.cohens_f2;
        if (beta) (*beta)(row, col) = result.beta;
        if (std_beta) (*std_beta)(row, col) = result.std_beta;
    };

    // Rows are continuous dependent variables, columns are categorical predictors followed by continuous
    // predictors. Pairs involving a confounder and self-regressions stay NaN.
    #pragma omp parallel for schedule(dynamic)
    for (size_t dependent_idx = 0; dependent_idx < num_cont_variables; ++dependent_idx)
    {
        const bool dependent_is_confounder = is_control_row(control_rows_continuous, dependent_idx);

        for (size_t independent_idx = 0; independent_idx < num_cat_variables; ++independent_idx)
        {
            RegressionResult result;
            if (!dependent_is_confounder && !is_control_row(control_rows_categorical, independent_idx))
            {
                result = pairwise_nan_linear_regression(
                    cat_data,
                    cont_data,
                    dependent_idx,
                    independent_idx,
                    control_rows_categorical,
                    control_rows_continuous,
                    na_value,
                    PredictorSource::Categorical);
            }
            store(dependent_idx, independent_idx, result);
        }

        for (size_t independent_idx = 0; independent_idx < num_cont_variables; ++independent_idx)
        {
            RegressionResult result;
            if (!dependent_is_confounder && independent_idx != dependent_idx &&
                !is_control_row(control_rows_continuous, independent_idx))
            {
                result = pairwise_nan_linear_regression(
                    cat_data,
                    cont_data,
                    dependent_idx,
                    independent_idx,
                    control_rows_categorical,
                    control_rows_continuous,
                    na_value,
                    PredictorSource::Continuous);
            }
            store(dependent_idx, num_cat_variables + independent_idx, result);
        }
    }

    return output;
}

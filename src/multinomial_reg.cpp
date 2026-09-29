#include "stats.hpp"
#include <algorithm>
#include <mlpack.hpp>

using namespace std;
using namespace mlpack;

enum class PredictorSource
{
    None,
    Categorical,
    Continuous
};

struct EncodedModel
{
    arma::mat predictors;
    arma::Row<size_t> labels;
    size_t num_classes = 0;
    size_t predictor_columns = 0;
};

std::vector<int> collect_valid_cols(
    const DataMatrix& cat_data,
    const DataMatrix& cont_data,
    int dependent_idx,
    int independent_idx,
    const std::vector<int>& control_rows_categorical,
    const std::vector<int>& control_rows_continuous,
    double na_value,
    PredictorSource predictor_source,
    bool include_predictor)
{
    std::vector<int> valid_cols;
    const size_t ncols = cat_data.cols();

    for (size_t c = 0; c < ncols; ++c) {
        bool valid = (cat_data(dependent_idx, c) != na_value);

        if (include_predictor) {
            if (predictor_source == PredictorSource::Categorical) {
                valid &= (cat_data(independent_idx, c) != na_value);
            } else if (predictor_source == PredictorSource::Continuous) {
                valid &= (cont_data(independent_idx, c) != na_value);
            }
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

std::vector<size_t> collect_levels(
    const DataMatrix& data,
    int row,
    const std::vector<int>& valid_cols,
    double na_value)
{
    std::vector<size_t> levels;

    for (int col : valid_cols) {
        const double value = data(row, static_cast<size_t>(col));
        if (value == na_value) {
            continue;
        }

        const size_t level = static_cast<size_t>(value);
        if (std::find(levels.begin(), levels.end(), level) == levels.end()) {
            levels.push_back(level);
        }
    }

    std::sort(levels.begin(), levels.end());
    return levels;
}

bool build_encoded_model(
    const DataMatrix& cat_data,
    const DataMatrix& cont_data,
    int dependent_idx,
    int independent_idx,
    const std::vector<int>& control_rows_categorical,
    const std::vector<int>& control_rows_continuous,
    double na_value,
    PredictorSource predictor_source,
    bool include_predictor,
    const std::vector<int>& valid_cols,
    EncodedModel& model)
{
    if (valid_cols.empty()) {
        return false;
    }

    const std::vector<size_t> label_levels = collect_levels(cat_data, dependent_idx, valid_cols, na_value);
    if (label_levels.size() < 2) {
        return false;
    }

    std::map<size_t, size_t> label_lookup;
    for (size_t i = 0; i < label_levels.size(); ++i) {
        label_lookup[label_levels[i]] = i;
    }

    size_t num_rows = control_rows_continuous.size();
    std::vector<std::pair<int, std::vector<size_t>>> categorical_rows;
    // Reserve space for control categorical rows and the predictor row if it's categorical
    categorical_rows.reserve(control_rows_categorical.size() + (include_predictor && predictor_source == PredictorSource::Categorical ? 1 : 0));

    for (int row : control_rows_categorical) {
        std::vector<size_t> levels = collect_levels(cat_data, row, valid_cols, na_value);
        if (levels.size() > 1) {
            num_rows += levels.size() - 1;
        }
        categorical_rows.push_back({row, std::move(levels)});
    }

    if (include_predictor && predictor_source == PredictorSource::Categorical) {
        std::vector<size_t> levels = collect_levels(cat_data, independent_idx, valid_cols, na_value);
        if (levels.size() > 1) {
            num_rows += levels.size() - 1;
        }
        model.predictor_columns = levels.size() > 1 ? levels.size() - 1 : 0;
        categorical_rows.push_back({independent_idx, std::move(levels)});
    } else if (include_predictor && predictor_source == PredictorSource::Continuous) {
        ++num_rows;
        model.predictor_columns = 1;
    } else {
        model.predictor_columns = 0;
    }

    model.predictors.zeros(num_rows, valid_cols.size());
    model.labels.set_size(valid_cols.size());
    model.num_classes = label_levels.size();

    for (size_t s = 0; s < valid_cols.size(); ++s) {
        const size_t label = static_cast<size_t>(cat_data(dependent_idx, static_cast<size_t>(valid_cols[s])));
        model.labels[s] = label_lookup[label];
    }

    size_t row_offset = 0;
    for (const auto& entry : categorical_rows) {
        const int source_row = entry.first;
        const std::vector<size_t>& levels = entry.second;
        if (levels.size() <= 1) {
            continue;
        }

        for (size_t level_idx = 1; level_idx < levels.size(); ++level_idx) {
            for (size_t s = 0; s < valid_cols.size(); ++s) {
                const size_t raw_level = static_cast<size_t>(cat_data(source_row, static_cast<size_t>(valid_cols[s])));
                model.predictors(row_offset, s) = (raw_level == levels[level_idx]) ? 1.0 : 0.0;
            }
            ++row_offset;
        }
    }

    for (int row : control_rows_continuous) {
        for (size_t s = 0; s < valid_cols.size(); ++s) {
            model.predictors(row_offset, s) = cont_data(row, static_cast<size_t>(valid_cols[s]));
        }
        ++row_offset;
    }

    if (include_predictor && predictor_source == PredictorSource::Continuous) {
        for (size_t s = 0; s < valid_cols.size(); ++s) {
            model.predictors(row_offset, s) = cont_data(independent_idx, static_cast<size_t>(valid_cols[s]));
        }
    }

    return true;
}

double fit_and_score(const EncodedModel& model)
{
    if (model.num_classes < 2 || model.labels.n_elem == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // mlpack cannot fit a model without features, but the maximum likelihood estimate of the intercept-only
    // model is given by the class frequencies.
    if (model.predictors.n_rows == 0) {
        std::vector<size_t> class_counts(model.num_classes, 0);
        for (size_t s = 0; s < model.labels.n_elem; ++s) {
            ++class_counts[model.labels[s]];
        }

        double log_likelihood = 0.0;
        for (size_t count : class_counts) {
            if (count > 0) {
                log_likelihood += count * std::log(static_cast<double>(count) / model.labels.n_elem);
            }
        }
        return log_likelihood;
    }

    // mlpack uses L2 regularization (lambda = 0.0001) by default, which biases the log-likelihood of the
    // likelihood-ratio test. Fit the unpenalized maximum likelihood model instead.
    mlpack::SoftmaxRegression<> regressor;
    regressor.Train(model.predictors, model.labels, model.num_classes, 0.0);

    arma::Row<size_t> predictions;
    arma::mat probabilities;
    regressor.Classify(model.predictors, predictions, probabilities);

    double log_likelihood = 0.0;
    for (size_t s = 0; s < model.labels.n_elem; ++s) {
        const double probability = std::max(probabilities(model.labels[s], s), 1e-300);
        log_likelihood += std::log(probability);
    }

    return log_likelihood;
}

std::pair<double, double> pairwise_nan_multinomial_regression(
    const DataMatrix& cat_data,
    const DataMatrix& cont_data,
    int dependent_idx,
    int independent_idx,
    const std::vector<int>& control_rows_categorical,
    const std::vector<int>& control_rows_continuous,
    double na_value,
    PredictorSource predictor_source)
{
    const std::vector<int> valid_cols = collect_valid_cols(
        cat_data,
        cont_data,
        dependent_idx,
        independent_idx,
        control_rows_categorical,
        control_rows_continuous,
        na_value,
        predictor_source,
        true);

    EncodedModel full_model;
    EncodedModel reduced_model;

    if (!build_encoded_model(
            cat_data,
            cont_data,
            dependent_idx,
            independent_idx,
            control_rows_categorical,
            control_rows_continuous,
            na_value,
            predictor_source,
            true,
            valid_cols,
            full_model)) {
        return {std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};
    }

    if (!build_encoded_model(
            cat_data,
            cont_data,
            dependent_idx,
            independent_idx,
            control_rows_categorical,
            control_rows_continuous,
            na_value,
            predictor_source,
            false,
            valid_cols,
            reduced_model)) {
        return {std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};
    }
    // Categorical predictor with a single remaining category adds no information to the reduced model.
    if (full_model.predictor_columns == 0) {
        return {std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};
    }

    const double log_likelihood_full = fit_and_score(full_model);
    const double log_likelihood_reduced = fit_and_score(reduced_model);

    if (std::isnan(log_likelihood_full) || std::isnan(log_likelihood_reduced)) {
        return {std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};
    }

    // The full model cannot fit worse than the reduced model; clamp numerical noise of the optimizer, since
    // boost throws on negative arguments, which must not happen inside an OpenMP region.
    const double lr_statistic = std::max(0.0, 2.0 * (log_likelihood_full - log_likelihood_reduced));
    const size_t degrees_of_freedom = (full_model.num_classes - 1) * full_model.predictor_columns;
    boost::math::chi_squared dist(degrees_of_freedom);
    const double p_value = cdf(complement(dist, lr_statistic));

    return std::make_pair(lr_statistic, p_value);
}

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

    #pragma omp parallel for
    for (size_t dependent_idx = 0; dependent_idx < num_cat_variables; ++dependent_idx)
    {
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

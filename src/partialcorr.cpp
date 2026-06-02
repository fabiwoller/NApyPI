#include <stats.hpp>
#include <eigen3/Eigen/Dense>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

VectorXd ols(const MatrixXd& X, const VectorXd y){
    return (X.transpose() * X).ldlt().solve(X.transpose() *y);
}

double pearsonCorrelation(const VectorXd& x, const VectorXd& y) {
    double mean_x = x.mean();
    double mean_y = y.mean();
    VectorXd diff_x = x.array() - mean_x;
    VectorXd diff_y = y.array() - mean_y;
    double numerator = diff_x.dot(diff_y);
    double denominator = std::sqrt(diff_x.squaredNorm() * diff_y.squaredNorm());
    return numerator / denominator;
    }

double spearmanCorrelation(const VectorXd& x, const VectorXd& y) {
    // Rank the data
    VectorXd rank_x = x;
    VectorXd rank_y = y;
    std::vector<std::pair<double, int>> pairs_x(x.size());
    std::vector<std::pair<double, int>> pairs_y(y.size());
    for (int i = 0; i < x.size(); ++i) {
        pairs_x[i] = {x(i), i};
        pairs_y[i] = {y(i), i};
    }
    std::sort(pairs_x.begin(), pairs_x.end());
    std::sort(pairs_y.begin(), pairs_y.end());
    for (int i = 0; i < x.size(); ++i) {
        rank_x(pairs_x[i].second) = i + 1;
        rank_y(pairs_y[i].second) = i + 1;
    }
    // Compute Pearson correlation on ranks
    return pearsonCorrelation(rank_x, rank_y);
}

VectorXd numeric_to_ranks(const VectorXd& vec) {
    const int n = static_cast<int>(vec.size());
    VectorXd ranks(n);
    std::vector<std::pair<double, int>> pairs(n);

    for (int i = 0; i < n; ++i) {
        pairs[i] = {vec(i), i};
    }
    std::sort(pairs.begin(), pairs.end());

    int i = 0;
    while (i < n) {
        // Finde das Ende der aktuellen Tie-Gruppe [i, j)
        int j = i + 1;
        while (j < n && pairs[j].first == pairs[i].first) {
            ++j;
        }
        // Midrank: Mittelwert der 1-basierten Ränge i+1 .. j
        const double mid = ( (i + 1) + j ) / 2.0;  // j ist exklusiv -> Endrang = j
        for (int k = i; k < j; ++k) {
            ranks(pairs[k].second) = mid;
        }
        i = j;
    }
    return ranks;
}

VectorXd residuals(const VectorXd& y, const MatrixXd& Z) {
    if (Z.cols() == 0) {
        // only intercept: residuals = y - mean(y)
        return y.array() - y.mean();
    }
    MatrixXd X(Z.rows(), Z.cols() + 1);
    X << VectorXd::Ones(Z.rows()), Z;
    VectorXd beta = ols(X, y);
    return y - X * beta;
}


std::pair<double, double> pairwise_nan_partial_correlation(
    const DataMatrix& data,
    int row1,
    int row2,
    const std::vector<int>& control_rows,
    double na_value, 
    const std::string& method)
{
    // check if row1 and row2 are in control_rows
    for (auto z : control_rows) {
        if (z == row1 || z == row2) {
            return {std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN()};
        }
    }


    std::vector<int> valid_cols;
    int ncols = data.cols();
    for (int c = 0; c < ncols; ++c) {
        bool valid = (data(row1, c) != na_value) && (data(row2, c) != na_value);
        for (auto z : control_rows)
            valid &= (data(z, c) != na_value);
        if (valid)
            valid_cols.push_back(c);
    }

    int n = valid_cols.size();
    if (n == 0) {
        return {std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};
    }

    VectorXd x(n), y(n);
    MatrixXd Z(n, control_rows.size());
    for (int i = 0; i < n; ++i) {
        int c = valid_cols[i];
        x(i) = data(row1, c);
        y(i) = data(row2, c);
        for (size_t j = 0; j < control_rows.size(); ++j)
            Z(i, j) = data(control_rows[j], c);
    }


    // check for constant control variables and drop them from Z
    std::vector<int> non_constant_indices;
    for (size_t j = 0; j < control_rows.size(); ++j) {
        VectorXd col = Z.col(j);
        if ((col.array() - col.mean()).abs().maxCoeff() != 0.0) {
            non_constant_indices.push_back(j);
        }
    }


    // reduce Z to non-constant columns
    MatrixXd Z_reduced(n, non_constant_indices.size());
    for (size_t j = 0; j < non_constant_indices.size(); ++j) {
        Z_reduced.col(j) = Z.col(non_constant_indices[j]);
    }
    Z = Z_reduced;

    //check for rank deficiency in Z
    if (Z.cols() > 0) {
        Eigen::ColPivHouseholderQR<MatrixXd> qr(Z);
        int rank = qr.rank();
        if (rank == 0) {
            // no explanatory variables (except intercept) -> residuals = y - mean(y)
            // we set Z to 0 columns to let residuals function recognize this
            Z.resize(Z.rows(), 0);
        } else if (rank < Z.cols()) {
            Eigen::VectorXi piv = qr.colsPermutation().indices();
            MatrixXd Z_indep(Z.rows(), rank);
            for (int k = 0; k < rank; ++k) {
                Z_indep.col(k) = Z.col(piv[k]);
            }
            Z = Z_indep;
        }
    }


    // if the method is spearman transform Z collumn-wise to ranks
    if (method == "spearman") {
        for (int j = 0; j < Z.cols(); ++j) {
            VectorXd col = Z.col(j);
            Z.col(j) = numeric_to_ranks(col);
        }
        x = numeric_to_ranks(x);
        y = numeric_to_ranks(y);
    }

    VectorXd resX = residuals(x, Z);
    VectorXd resY = residuals(y, Z);

    // Compute correlation of residuals
    double r = pearsonCorrelation(resX, resY);

    // Compute p-value
    int df = n - static_cast<int>(non_constant_indices.size()) - 2;
    if (df > 0) {
        double denom = 1.0 - r * r;
        double t;
        if (denom == 0) {
            t = std::copysign(std::numeric_limits<double>::infinity(), r);
        } else {
            t = r * std::sqrt(static_cast<double>(df) / denom);
        }
        boost::math::students_t dist(df);
        double p = 2 * boost::math::cdf(dist, -std::abs(t));
        return std::make_pair(r, p);
    }
    else {
        return std::make_pair(r, std::numeric_limits<double>::quiet_NaN());
    }
}

std::pair<DataMatrix, DataMatrix> statistics::partial_correlation_with_nans(
    const DataMatrix& data, 
    const std::vector<int>& control_rows,
    double na_value, 
    const std::string& method)
{
    const int num_rows = data.rows();
    DataMatrix correlations(num_rows, num_rows);
    DataMatrix pvalues(num_rows, num_rows);

    // Compute pairwise partial correlations for all rows in DataMatrix.
    #pragma omp parallel for
    for (int iR = 0; iR < num_rows; ++iR)
    {
        for (int jR = iR; jR < num_rows; ++jR)
        {
            std::pair<double, double> results = pairwise_nan_partial_correlation(data, iR, jR, control_rows, na_value, method);
            correlations(iR, jR) = std::get<0>(results);
            correlations(jR, iR) = std::get<0>(results);
            double pvalue = std::get<1>(results);
            pvalues(iR, jR) = pvalue;
            pvalues(jR, iR) = pvalue;
        }
    }

    return std::make_pair(correlations, pvalues);
}

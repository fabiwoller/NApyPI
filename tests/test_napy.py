import napypi as napy
import numpy as np
import scipy as sc
import pandas as pd
import pingouin as pg
import unittest
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import Formula
import os

USE_NUMBA = os.getenv("USE_NUMBA", "0") == "1"

class TestPearsonCorrelation(unittest.TestCase):
    
    def test_basic(self):
        """Test basic correlation computation against scipy.
        """
        data = np.array([[1,2,3,4], [2,4,3,3]])
        nan_value = -99
        out_dict = napy.pearsonr(data, nan_value=nan_value, axis=0, threads=1, use_numba=USE_NUMBA)
        napy_corr = out_dict['r']
        napy_pvals = out_dict['p_unadjusted']
        scipy_corr, scipy_pvals = sc.stats.pearsonr(data[0], data[1])
        self.assertEqual(napy_corr[0,0], 1.0)
        self.assertEqual(napy_pvals[0,0], 0.0)
        self.assertAlmostEqual(napy_corr[0,1], scipy_corr)
        self.assertAlmostEqual(napy_pvals[0,1], scipy_pvals)
        self.assertAlmostEqual(napy_corr[0,1], napy_corr[1,0])
        self.assertAlmostEqual(napy_pvals[0,1], napy_pvals[1,0])
        
    def test_large(self):
        """Test correlation results on larger input data.
        """
        nan_value = -99
        data = np.random.rand(2, 100)
        out_dict = napy.pearsonr(data, nan_value=nan_value, axis=0, threads=1, use_numba=USE_NUMBA)
        corr1 = out_dict['r']
        pval1 = out_dict['p_unadjusted']
        corr2, pval2 = sc.stats.pearsonr(data[0], data[1])
        self.assertAlmostEqual(corr1[0,1], corr2)
        self.assertAlmostEqual(pval1[0,1], pval2)
    
    def test_nans(self):
        """Test correlation functionality under NA existence.
        """
        nan_value = -99
        data = np.array([[1,2,3,4,5,nan_value], [nan_value,2,5,4,nan_value,6]])
        out_dict= napy.pearsonr(data, nan_value=nan_value, axis=0, threads=1, use_numba=USE_NUMBA)
        napy_corr = out_dict['r']
        napy_pvals = out_dict['p_unadjusted']
        scipy_corr, scipy_pvals = sc.stats.pearsonr([2,3,4], [2,5,4])
        self.assertAlmostEqual(napy_corr[0,1], scipy_corr)
        self.assertAlmostEqual(napy_pvals[1,0], scipy_pvals)
        
    def test_axis(self):
        """Test axis parameter of pearson computation.
        """
        nan_value = -99
        data = np.random.rand(3,3)
        data_T = data.T.copy()
        out_dict1 = napy.pearsonr(data, nan_value=nan_value, axis=0, threads=1, use_numba=USE_NUMBA)
        napy_corr1 = out_dict1['r']
        napy_pvals1 = out_dict1['p_unadjusted']
        out_dict2 = napy.pearsonr(data_T, nan_value=nan_value, axis=1, threads=1, use_numba=USE_NUMBA)
        napy_corr2 = out_dict2['r']
        napy_pvals2 = out_dict2['p_unadjusted']
        self.assertListEqual(napy_corr1.tolist(), napy_corr2.tolist())
        self.assertListEqual(napy_pvals1.tolist(), napy_pvals2.tolist())
        
    def test_parallel(self):
        """Test parallel execution results.
        """
        nan_value = -99
        data = np.random.rand(10, 5)
        out_dict1 = napy.pearsonr(data, nan_value=nan_value, axis=0, threads=1, use_numba=USE_NUMBA)
        corr1 = out_dict1['r']
        pval1 = out_dict1['p_unadjusted']
        out_dict2 = napy.pearsonr(data, nan_value=nan_value, axis=0, threads=4, use_numba=USE_NUMBA)
        corr2 = out_dict2['r']
        pval2 = out_dict2['p_unadjusted']
        self.assertListEqual(corr1.tolist(), corr2.tolist())
        self.assertListEqual(pval1.tolist(), pval2.tolist())
        
    def test_pvalue(self):
        """Test edge cases of pvalue computation.
        """
        nan_value = -99
        data = np.array([[1,2,nan_value], [2,3,4]])
        out_dict = napy.pearsonr(data, nan_value=nan_value, axis=0, threads=1, use_numba=USE_NUMBA)
        pvals = out_dict['p_unadjusted']
        self.assertTrue(np.isnan(pvals[0,0]))
        self.assertTrue(np.isnan(pvals[0,1]))
        
    def test_short_input(self):
        """Test edge case of input length one.
        """
        nan_value = -99
        data = np.array([[1,2,nan_value], [nan_value, 2, 3]])
        out_dict = napy.pearsonr(data, nan_value=nan_value, axis=0, threads=1, use_numba=USE_NUMBA)
        corrs = out_dict['r']
        pvals = out_dict['p_unadjusted']
        self.assertTrue(np.isnan(corrs[0,1]))
        self.assertTrue(np.isnan(corrs[1,0]))
        self.assertTrue(np.isnan(pvals[0,0]))
        self.assertTrue(np.isnan(pvals[1,0]))
        
    def test_empty_input(self):
        """Test case when input is empty due to all NA removal.
        """
        nan_value = -99
        data = np.array([[-99, 1, 2, -99], [3,-99, -99, 1]])
        out_dict = napy.pearsonr(data, nan_value=nan_value, axis=0, threads=1, use_numba=USE_NUMBA)
        corrs = out_dict['r']
        pvals = out_dict['p_unadjusted']
        self.assertTrue(np.isnan(corrs[0,1]))
        self.assertTrue(np.isnan(pvals[0,1]))
        self.assertTrue(np.isnan(corrs[1,0]))
        self.assertTrue(np.isnan(pvals[1,0]))
        
    def test_basic_R(self):
        """Basic test case against R library function.
        """
        numpy2ri.activate()
        data = np.random.rand(2, 10)

        # Convert the NumPy arrays to R vectors
        r_x = numpy2ri.py2rpy(data[0, :])
        r_y = numpy2ri.py2rpy(data[1, :])

        # Call the R cor function
        cor = robjects.r['cor.test']
        correlation = cor(r_x, r_y)

        # Print the result
        py_result = dict(zip(correlation.names, map(list,list(correlation))))
        r_pvalue = py_result['p.value'][0]
        r_corr = py_result['estimate'][0]
        
        out_dict = napy.pearsonr(data, use_numba=USE_NUMBA)
        napy_corr = out_dict['r']
        napy_pvals = out_dict['p_unadjusted']
        self.assertAlmostEqual(r_corr, napy_corr[0,1])
        self.assertAlmostEqual(r_pvalue, napy_pvals[0,1])
        self.assertAlmostEqual(r_corr, napy_corr[1,0])
        self.assertAlmostEqual(r_pvalue, napy_pvals[1,0])


class TestSpearmanCorrelation(unittest.TestCase):
    
    def test_basic(self):
        """Test basic correlation computation against scipy.
        """
        data = np.array([[1,2,3,4], [2,4,3,3]])
        nan_value = -99
        out_dict = napy.spearmanr(data, nan_value=nan_value, axis=0, threads=1, use_numba=USE_NUMBA)
        napy_corr = out_dict['rho']
        napy_pvals = out_dict['p_unadjusted']
        scipy_corr, scipy_pvals = sc.stats.spearmanr(data[0], data[1])
        self.assertEqual(napy_corr[0,0], 1.0)
        self.assertEqual(napy_pvals[0,0], 0.0)
        self.assertAlmostEqual(napy_corr[0,1], scipy_corr)
        self.assertAlmostEqual(napy_pvals[0,1], scipy_pvals)
        self.assertAlmostEqual(napy_corr[0,1], napy_corr[1,0])
        self.assertAlmostEqual(napy_pvals[0,1], napy_pvals[1,0])

    def test_large(self):
        """Test correlation results on larger input data.
        """
        nan_value = -99
        data = np.random.rand(2, 100)
        out_dict = napy.spearmanr(data, nan_value=nan_value, axis=0, threads=1, use_numba=USE_NUMBA)
        corr1 = out_dict['rho']
        pval1 = out_dict['p_unadjusted']
        corr2, pval2 = sc.stats.spearmanr(data[0], data[1])
        self.assertAlmostEqual(corr1[0,1], corr2)
        self.assertAlmostEqual(pval1[0,1], pval2)

    def test_nans(self):
        """Test correlation functionality under NA existence.
        """
        nan_value = -99
        data = np.array([[1,2,3,4,5,nan_value], [nan_value,2,5,4,nan_value,6]])
        out_dict = napy.spearmanr(data, nan_value=nan_value, axis=0, threads=1, use_numba=USE_NUMBA)
        napy_corr = out_dict['rho']
        napy_pvals = out_dict['p_unadjusted']
        scipy_corr, scipy_pvals = sc.stats.spearmanr([2,3,4], [2,5,4])
        self.assertAlmostEqual(napy_corr[0,1], scipy_corr)
        self.assertAlmostEqual(napy_pvals[1,0], scipy_pvals)
        
    def test_axis(self):
        """Test axis parameter of spearmanr computation.
        """
        nan_value = -99
        data = np.random.rand(3,3)
        data_T = data.copy().T
        out_dict1 = napy.spearmanr(data, nan_value=nan_value, axis=0, threads=1, use_numba=USE_NUMBA)
        napy_corr1 = out_dict1['rho']
        napy_pvals1 = out_dict1['p_unadjusted']
        out_dict2 = napy.spearmanr(data_T, nan_value=nan_value, axis=1, threads=1, use_numba=USE_NUMBA)
        napy_corr2 = out_dict2['rho']
        napy_pvals2 = out_dict2['p_unadjusted']
        self.assertListEqual(napy_corr1.tolist(), napy_corr2.tolist())
        self.assertListEqual(napy_pvals1.tolist(), napy_pvals2.tolist())
        
    def test_parallel(self):
        """Test parallel execution results.
        """
        nan_value = -99
        data = np.random.rand(10, 5)
        out_dict1 = napy.spearmanr(data, nan_value=nan_value, axis=0, threads=1, use_numba=USE_NUMBA)
        corr1 = out_dict1['rho']
        pval1 = out_dict1['p_unadjusted']
        out_dict2 = napy.spearmanr(data, nan_value=nan_value, axis=0, threads=4, use_numba=USE_NUMBA)
        corr2 = out_dict2['rho']
        pval2 = out_dict2['p_unadjusted']
        self.assertListEqual(corr1.tolist(), corr2.tolist())
        self.assertListEqual(pval1.tolist(), pval2.tolist())
        
    def test_pvalue(self):
        """Test edge cases of pvalue computation.
        """
        nan_value = -99
        data = np.array([[1,2,nan_value], [2,3,4]])
        out_dict = napy.spearmanr(data, nan_value=nan_value, axis=0, threads=1, use_numba=USE_NUMBA)
        corrs = out_dict['rho']
        pvals = out_dict['p_unadjusted']
        self.assertTrue(np.isnan(pvals[0,0]))
        self.assertTrue(np.isnan(pvals[0,1]))
        
    def test_short_input(self):
        """Test edge case of input length one.
        """
        nan_value = -99
        data = np.array([[1,2,nan_value], [nan_value, 2, 3]])
        out_dict = napy.spearmanr(data, nan_value=nan_value, axis=0, threads=1, use_numba=USE_NUMBA)
        corrs = out_dict['rho']
        pvals = out_dict['p_unadjusted']
        self.assertTrue(np.isnan(corrs[0,1]))
        self.assertTrue(np.isnan(corrs[1,0]))
        self.assertTrue(np.isnan(pvals[0,0]))
        self.assertTrue(np.isnan(pvals[1,0]))
        
    def test_basic_R(self):
        """Basic test case against R library function.
        """
        numpy2ri.activate()
        hmisc = importr('Hmisc')
        data = np.random.rand(2, 100)

        # Convert the NumPy arrays to R vectors
        r_x = numpy2ri.py2rpy(data[0, :])
        r_y = numpy2ri.py2rpy(data[1, :])

        # Call the R cor function
        rcorr_res = hmisc.rcorr(r_x, r_y, type='spearman')
        r_p_values = rcorr_res.rx2('P')
        r_correlations = rcorr_res.rx2('r')

        out_dict = napy.spearmanr(data, use_numba=USE_NUMBA)
        napy_corr = out_dict['rho']
        napy_pvals = out_dict['p_unadjusted']
        self.assertAlmostEqual(r_correlations[0,1], napy_corr[0,1])
        self.assertAlmostEqual(r_correlations[1,0], napy_corr[1,0])
        self.assertAlmostEqual(r_p_values[0,1], napy_pvals[0,1])
        self.assertAlmostEqual(r_p_values[1,0], napy_pvals[1,0])

class TestChiSquared(unittest.TestCase):

    def test_basic(self):
        """Test basic functionality against scipy implementation.
        """
        data = np.array([[0,1,1,1,0], [2,0,1,0,1]])
        nan_value = -99
        cont_table = np.array([[0,1,1], [2,1,0]])
        scipy_result = sc.stats.chi2_contingency(cont_table, correction=False)
        cont_table_self = np.array([[2,0], [0,3]])
        scipy_result_self = sc.stats.chi2_contingency(cont_table_self, correction=False)
        out_dict = napy.chi_squared(data, nan_value=nan_value, use_numba=USE_NUMBA)
        stats = out_dict['chi2']
        pvalues = out_dict['p_unadjusted']
        self.assertAlmostEqual(stats[0,1], scipy_result.statistic)
        self.assertAlmostEqual(pvalues[0,1], scipy_result.pvalue)
        self.assertAlmostEqual(stats[1,0], scipy_result.statistic)
        self.assertAlmostEqual(pvalues[1,0], scipy_result.pvalue)
        self.assertAlmostEqual(stats[0,0], scipy_result_self.statistic)
        self.assertAlmostEqual(pvalues[0,0], scipy_result_self.pvalue)

    def test_parallel(self):
        """Test parallel correctness.
        """
        cat_data = np.array([[0,0,1,1,1], [2,1,1,0,2], [0,0,0,1,1], [2,1,0,0,0]])
        out_dict1 = napy.chi_squared(cat_data, threads=1, use_numba=USE_NUMBA)
        s = out_dict1['chi2']
        p = out_dict1['p_unadjusted']
        out_dict2 = napy.chi_squared(cat_data, threads=2, use_numba=USE_NUMBA)
        s_par = out_dict2['chi2']
        p_par = out_dict2['p_unadjusted']
        self.assertListEqual(s.tolist(), s_par.tolist())
        self.assertListEqual(p.tolist(), p_par.tolist())

    def test_axis_parameter(self):
        cat_data = np.array([[1,0,1,1], [0,1,0,1], [1,0,0,0], [0,1,0,1]])
        cat_data_T = cat_data.T.copy()
        out_dict = napy.chi_squared(cat_data, check_data=True, use_numba=USE_NUMBA)
        s = out_dict['chi2']
        p = out_dict['p_unadjusted']
        out_dict_t = napy.chi_squared(cat_data_T, axis=1, check_data=True, use_numba=USE_NUMBA)
        s_t = out_dict_t['chi2']
        p_t = out_dict_t['p_unadjusted']
        self.assertListEqual(s.tolist(), s_t.tolist())
        self.assertListEqual(p.tolist(), p_t.tolist())

    def test_na(self):
        """Test na functionality of chi squared implementation.
        """
        nan_value = -99
        data = np.array([[0,1,1,-99,0], [2,0,1,0,-99]])
        cont_table = np.array([[0,0,1], [1,1,0]])
        scipy_res = sc.stats.chi2_contingency(cont_table, correction=False)
        out_dict = napy.chi_squared(data, nan_value=nan_value, use_numba=USE_NUMBA)
        stats = out_dict['chi2']
        pvalues = out_dict['p_unadjusted']
        self.assertAlmostEqual(stats[0,1], scipy_res.statistic)
        self.assertAlmostEqual(pvalues[0,1], scipy_res.pvalue)

    def test_single_category(self):
        """Test functionality if only one category present in one variable.
        """
        nan_value = -99
        data = np.array([[0,0,0], [0,1,2]])
        out_dict = napy.chi_squared(data, nan_value=nan_value, use_numba=USE_NUMBA)
        s = out_dict['chi2']
        p = out_dict['p_unadjusted']
        self.assertAlmostEqual(s[0,1], 0.0)
        self.assertTrue(np.isnan(p[0,1]))
        self.assertAlmostEqual(s[1,0], 0.0)
        self.assertAlmostEqual(s[0,0], 0.0)
        self.assertTrue(np.isnan(p[1,0]))

    def test_wrong_input(self):
        """Test if exception is raised when check_data parameter is used.
        """
        data = np.array([[1,1,1,1,2], [0,2,1,0,0]])
        with self.assertRaises(ValueError):
            napy.chi_squared(data, check_data=True, use_numba=USE_NUMBA)
        data = np.array([[0,1,2,1,1], [3,0,1,1,1]])
        with self.assertRaises(ValueError):
            napy.chi_squared(data, check_data=True, use_numba=USE_NUMBA)

    def test_missing_category(self):
        na = -99
        data = np.array([[0,1,1,0,na], [na,1,0,2,3]])
        out_dict = napy.chi_squared(data, nan_value=na, use_numba=USE_NUMBA)
        s = out_dict['chi2']
        p = out_dict['p_unadjusted']
        self.assertTrue(np.isnan(s[0,1]))
        self.assertTrue(np.isnan(s[1,0]))
        self.assertTrue(np.isnan(p[0,1]))
        self.assertTrue(np.isnan(p[1,0]))

    def test_no_values_with_na(self):
        """Test case in which no pairwise non-NA values exist.
        """
        data = np.array([[-99, 0, -99, 1], [0, -99, 1, -99]])
        out_dict = napy.chi_squared(data, nan_value=-99, use_numba=USE_NUMBA)
        s = out_dict['chi2']
        p = out_dict['p_unadjusted']
        self.assertTrue(np.isnan(s[0,1]))
        self.assertTrue(np.isnan(s[1,0]))
        self.assertTrue(np.isnan(p[0,1]))
        self.assertTrue(np.isnan(p[1,0]))

    def test_basic_R(self):
        """Test case for comparison against R implementation.
        """
        numpy2ri.activate()
        stats = importr('stats')
        data = np.array([[1,1,0,1,0,0,0], [2,1,1,0,2,0,2]])

        # Convert the NumPy arrays to R vectors
        r_x = numpy2ri.py2rpy(data[0, :])
        r_y = numpy2ri.py2rpy(data[1, :])

        # Call the R cor function
        chisq_res = stats.chisq_test(r_x, r_y)
        p_value = chisq_res.rx2('p.value')[0]
        test_statistic = chisq_res.rx2('statistic')[0]

        out_dict = napy.chi_squared(data, use_numba=USE_NUMBA)
        napy_chi2 = out_dict['chi2']
        napy_pvals = out_dict['p_unadjusted']
        self.assertAlmostEqual(test_statistic, napy_chi2[0,1])
        self.assertAlmostEqual(test_statistic, napy_chi2[1,0])
        self.assertAlmostEqual(p_value, napy_pvals[0,1])
        self.assertAlmostEqual(p_value, napy_pvals[1,0])

    def test_effect_size_phi(self):
        """
        Test calculation of phi effect size against R implementation.
        """
        numpy2ri.activate()
        stats = importr('stats')
        effectsize_r = importr('effectsize')
        data = np.array([[1, 1, 0, 1, 0, 0, 0], [2, 1, 1, 0, 2, 0, 2]])

        # Convert the NumPy arrays to R vectors
        r_x = numpy2ri.py2rpy(data[0, :])
        r_y = numpy2ri.py2rpy(data[1, :])

        # Call the R cor function
        chisq_res = stats.chisq_test(r_x, r_y)
        x2_statistic = chisq_res.rx2('statistic')[0]

        # Call effect size function in R.
        phi_eff_size = effectsize_r.chisq_to_phi(x2_statistic, 7, 2, 2, adjust=False)
        phi = phi_eff_size.rx(1)[0][0]

        out_dict = napy.chi_squared(data, use_numba=USE_NUMBA)
        napy_phi = out_dict['phi']
        self.assertAlmostEqual(phi, napy_phi[0, 1])
        self.assertAlmostEqual(phi, napy_phi[1, 0])

    def test_effect_size_v(self):
        """
        Test Cramer's V effect size against R implementation.
        """
        numpy2ri.activate()
        stats = importr('stats')
        effectsize_r = importr('effectsize')
        data = np.array([[1, 1, 0, 1, 0, 0, 0], [2, 1, 1, 0, 2, 0, 2]])

        # Convert the NumPy arrays to R vectors
        r_x = numpy2ri.py2rpy(data[0, :])
        r_y = numpy2ri.py2rpy(data[1, :])

        # Call the R cor function
        chisq_res = stats.chisq_test(r_x, r_y)
        x2_statistic = chisq_res.rx2('statistic')[0]

        # Call effect size function in R.
        cramer_eff_size = effectsize_r.chisq_to_cramers_v(x2_statistic, 7, 2, 3, adjust=False)
        cramer = cramer_eff_size.rx(1)[0][0]

        out_dict = napy.chi_squared(data, use_numba=USE_NUMBA)
        napy_cramer = out_dict['cramers_v']
        self.assertAlmostEqual(cramer, napy_cramer[0, 1])
        self.assertAlmostEqual(cramer, napy_cramer[1, 0])


class TestANOVA(unittest.TestCase):
    
    def test_scipy(self):
        """Basic testing against scipy implementation.
        """
        cat_data = np.array([[0,1,1,1,0], [1,2,2,1,0]])
        cont_data = np.array([[4,4,1,2,1], [0,0,1,5,1]])
        out_dict = napy.anova(cat_data, cont_data, use_numba=USE_NUMBA)
        napy_stats = out_dict['F']
        napy_pvals = out_dict['p_unadjusted']
        stat00, pval00 = sc.stats.f_oneway([4,1], [4,1,2])
        stat01, pval01 = sc.stats.f_oneway([0,1], [0,1,5])
        stat11, pval11 = sc.stats.f_oneway([1], [0,5], [0,1])
        stat10, pval10 = sc.stats.f_oneway([1], [4,2], [4,1])
        self.assertAlmostEqual(napy_stats[0,0], stat00)
        self.assertAlmostEqual(napy_pvals[0,0], pval00)
        self.assertAlmostEqual(napy_stats[0,1], stat01)
        self.assertAlmostEqual(napy_pvals[0,1], pval01)
        self.assertAlmostEqual(napy_stats[1,1], stat11)
        self.assertAlmostEqual(napy_pvals[1,1], pval11)
        self.assertAlmostEqual(napy_stats[1,0], stat10)
        self.assertAlmostEqual(napy_pvals[1,0], pval10)
        
    def test_ill_defined_dofs(self):
        """Test case for ill-defined DOFs in ANOVA.
        """
        cat_data = np.array([[0,1,2,3]])
        cont_data = np.array([[3,2,5,5]])
        out_dict = napy.anova(cat_data, cont_data, use_numba=USE_NUMBA)
        s = out_dict['F']
        p = out_dict['p_unadjusted']
        self.assertTrue(np.isnan(s[0]))
        self.assertTrue(np.isnan(p[0]))

    def test_empty_category_without_ignore(self):
        """Test case for non-existing category after nan removal.
        """
        cat_data = np.array([[0,2,1,0,2,1]])
        cont_data = np.array([[-99, 4, 4,-99,2,3]])
        out_dict = napy.anova(cat_data, cont_data, nan_value=-99, ignore_empty_groups=False, use_numba=USE_NUMBA)
        s = out_dict['F']
        p = out_dict['p_unadjusted']
        self.assertTrue(np.isnan(s[0]))
        self.assertTrue(np.isnan(p[0]))
        
    def test_empty_category_with_ignore(self):
        """Test case for non-existing category after nan removal.
        """
        cat_data = np.array([[0,2,1,0,2,1]])
        cont_data = np.array([[-99, 4, 4,-99,2,3]])
        out_dict = napy.anova(cat_data, cont_data, nan_value=-99, ignore_empty_groups=True, use_numba=USE_NUMBA)
        s_napy = out_dict['F']
        p_napy = out_dict['p_unadjusted']
        s_scipy, pval_scipy = sc.stats.f_oneway([4,3], [4,2])
        self.assertAlmostEqual(s_napy[0,0], s_scipy)
        self.assertAlmostEqual(p_napy[0,0], pval_scipy)
        
    def test_all_nan_removal(self):
        """Test case if no elements exist anymore after NA removal.
        """
        cat_data = np.array([[-99, 0, 1, -99]])
        cont_data = np.array([[3, -99, -99, 2]])
        out_dict = napy.anova(cat_data, cont_data, nan_value=-99, use_numba=USE_NUMBA)
        s = out_dict['F']
        p = out_dict['p_unadjusted']
        self.assertTrue(np.isnan(s[0]))
        self.assertTrue(np.isnan(p[0]))
    
    def test_constant_input(self):
        """Test if all input measurements are zero in all groups.
        """
        cat_data = np.array([[0,2,2,1,3]])
        cont_data = np.array([[0,0,0,0,0]])
        out_dict = napy.anova(cat_data, cont_data, use_numba=USE_NUMBA)
        s = out_dict['F']
        p = out_dict['p_unadjusted']
        self.assertTrue(np.isnan(s[0]))
        self.assertTrue(np.isnan(p[0]))
        
    def test_zero_variances(self):
        """Test if categories are all constant (against scipy).
        """
        cat_data = np.array([[0,0,1,1]])
        cont_data = np.array([[1,1,2,2]])
        out_dict = napy.anova(cat_data, cont_data, use_numba=USE_NUMBA)
        napy_s = out_dict['F']
        napy_p = out_dict['p_unadjusted']
        scipy_s, scipy_p = sc.stats.f_oneway([1,1], [2,2])
        self.assertEqual(napy_s[0], scipy_s)
        self.assertEqual(napy_p[0], scipy_p)
        
    def test_threads(self):
        """Test on parallel correctness.
        """
        cat_data = np.array([[2,2,0,1], [1,1,0,1], [0,1,2,2]])
        cont_data = np.array([[2,3,3,4], [0,0,1,0], [3,3,0,1]])
        out_dict1 = napy.anova(cat_data, cont_data, threads=1, use_numba=USE_NUMBA)
        s1 = out_dict1['F']
        p1 = out_dict1['p_unadjusted']
        out_dict4 = napy.anova(cat_data, cont_data, threads=4, use_numba=USE_NUMBA)
        s4 = out_dict4['F']
        p4 = out_dict4['p_unadjusted']
        self.assertListEqual(s1.tolist(), s4.tolist())
        self.assertListEqual(p1.tolist(), p4.tolist())
    
    def test_axis_param(self):
        """Test axis parameter of anova function.
        """
        cat_data = np.eye(3)
        cont_data = np.random.rand(3,3)
        out_dict = napy.anova(cat_data, cont_data, use_numba=USE_NUMBA)
        s = out_dict['F']
        p = out_dict['p_unadjusted']
        cont_transp = cont_data.T.copy()
        out_dict_t = napy.anova(cat_data, cont_transp, axis=1, use_numba=USE_NUMBA)
        s_t = out_dict_t['F']
        p_t = out_dict_t['p_unadjusted']
        self.assertListEqual(s.tolist(), s_t.tolist())
        self.assertListEqual(p.tolist(), p_t.tolist())
        
    def test_wrong_input(self):
        """Test if exception is raised when check_data parameter is used.
        """
        cont_data = np.random.rand(2,5)
        cat_data = np.array([[1,1,1,1,2], [0,2,1,0,0]])
        with self.assertRaises(ValueError):
            napy.anova(cat_data, cont_data, check_data=True, use_numba=USE_NUMBA)
        cat_data = np.array([[0,1,2,1,1], [3,0,1,1,1]])
        with self.assertRaises(ValueError):
            napy.anova(cat_data, cont_data, check_data=True, use_numba=USE_NUMBA)

    def test_na_removal(self):
        """Test basic NA removal.
        """
        cat_data1 = np.array([[0,    0,-99,1,2,0]])
        cont_data1 = np.array([[-99, 3,3,  5,4,1]])
        out_dict1 = napy.anova(cat_data1, cont_data1, nan_value=-99, use_numba=USE_NUMBA)
        s1 = out_dict1['F']
        p1 = out_dict1['p_unadjusted']
        cat_data_clean = np.array([[0,1,2,0]])
        cont_data_clean = np.array([[3,5,4,1]])
        out_dict_clean = napy.anova(cat_data_clean, cont_data_clean, use_numba=USE_NUMBA)
        s_clean = out_dict_clean['F']
        p_clean = out_dict_clean['p_unadjusted']
        self.assertListEqual(s1.tolist(), s_clean.tolist())
        self.assertListEqual(p1.tolist(), p_clean.tolist())
    
    def test_basic_R(self):
        """Test basic functionality against R implementation.
        """
        numpy2ri.activate()
        pandas2ri.activate()
        
        stats = importr('stats')
        cat_data = [1,0,2,1,2,0]
        cont_data = [4,1,1,0,5,2]
        data = pd.DataFrame({'continuous': cont_data, 'categorical': cat_data})
        napy_cat = np.array([[1,0,2,1,2,0]])
        napy_cont = np.array([[4,1,1,0,5,2]])

        # Convert the NumPy arrays to R vectors
        r_data = pandas2ri.py2rpy(data)
        cat_index = list(r_data.colnames).index('categorical')
        cat_col = robjects.vectors.FactorVector(r_data.rx2('categorical'))
        r_data[cat_index] = cat_col
        formula = Formula("continuous ~ categorical")

        # Call the R cor function
        anova_res = stats.aov(formula, data=r_data)
        summary_res = robjects.r['summary'](anova_res)
        
        f_statistic = summary_res.rx2(1).rx2("F value")[0]
        p_value = summary_res.rx2(1).rx2("Pr(>F)")[0]

        out_dict = napy.anova(napy_cat, napy_cont, use_numba=USE_NUMBA)
        napy_corr = out_dict['F']
        napy_pvals = out_dict['p_unadjusted']
        self.assertAlmostEqual(f_statistic, napy_corr[0,0])
        self.assertAlmostEqual(f_statistic, napy_corr[0,0])
        self.assertAlmostEqual(p_value, napy_pvals[0,0])
        self.assertAlmostEqual(p_value, napy_pvals[0,0])

    def test_partial_eta2(self):
        """
        Test calculation of partial eta squared effect size.
        """
        numpy2ri.activate()
        pandas2ri.activate()

        stats = importr('stats')
        rstatix = importr('rstatix')
        cat_data = [1, 0, 2, 1, 2, 0]
        cont_data = [4, 1, 1, 0, 5, 2]
        data = pd.DataFrame({'continuous': cont_data, 'categorical': cat_data})
        napy_cat = np.array([[1, 0, 2, 1, 2, 0]])
        napy_cont = np.array([[4, 1, 1, 0, 5, 2]])

        # Convert the NumPy arrays to R vectors
        r_data = pandas2ri.py2rpy(data)
        cat_index = list(r_data.colnames).index('categorical')
        cat_col = robjects.vectors.FactorVector(r_data.rx2('categorical'))
        r_data[cat_index] = cat_col
        formula = Formula("continuous ~ categorical")

        # Call the R aov function
        anova_res = stats.aov(formula, data=r_data)

        # Compute effect size in R
        np2_r = rstatix.partial_eta_squared(anova_res)[0]

        # Compare against napy.
        out_dict = napy.anova(napy_cat, napy_cont, use_numba=USE_NUMBA)
        np2_mat = out_dict['np2']
        self.assertAlmostEqual(np2_r, np2_mat[0,0])

        
class TestKruskalWallis(unittest.TestCase):
    
    def test_basic(self):
        """Basic testing against scipy implementation.
        """
        cat_data = np.array([[2,1,1,0,1], [1,1,0,1,0]])
        cont_data = np.array([[1,4,3,3,1], [3,3,1,3,2]])
        out_dict = napy.kruskal_wallis(cat_data, cont_data, use_numba=USE_NUMBA)
        s_napy = out_dict['H']
        p_napy = out_dict['p_unadjusted']
        s00, p00 = sc.stats.kruskal([3], [1,3,4], [1])
        s11, p11 = sc.stats.kruskal([1,2], [3,3,3])
        s01, p01 = sc.stats.kruskal([3], [3], [3,1,2])
        self.assertAlmostEqual(s00, s_napy[0,0])
        self.assertAlmostEqual(p00, p_napy[0,0])
        self.assertAlmostEqual(s11, s_napy[1,1])
        self.assertAlmostEqual(p11, p_napy[1,1])
        self.assertAlmostEqual(s01, s_napy[0,1])
        self.assertAlmostEqual(p01, p_napy[0,1])
        
    def test_na_removal(self):
        """Test NA removal strategy.
        """
        cat_data = np.array([[0,2,-99,2,1,1,2]])
        cont_data = np.array([[4,-99,4,6,7,3,3]])
        out_dict = napy.kruskal_wallis(cat_data, cont_data, nan_value=-99, use_numba=USE_NUMBA)
        s_napy = out_dict['H']
        p_napy = out_dict['p_unadjusted']
        s_sc, p_sc = sc.stats.kruskal([4], [3,7], [3,6])
        self.assertAlmostEqual(s_napy[0], s_sc)
        self.assertAlmostEqual(p_napy[0], p_sc)
        
    def test_axis_param(self):
        """Test functionality of axis input parameter.
        """
        cat_data = np.eye(3)
        cont_data = np.random.rand(3,3)
        out_dic = napy.kruskal_wallis(cat_data, cont_data, use_numba=USE_NUMBA)
        s = out_dic['H']
        p = out_dic['p_unadjusted']
        cont_transp = cont_data.T.copy()
        out_dic_t = napy.kruskal_wallis(cat_data, cont_transp, axis=1, use_numba=USE_NUMBA)
        s_t = out_dic_t['H']
        p_t = out_dic_t['p_unadjusted']
        self.assertListEqual(s.tolist(), s_t.tolist())
        self.assertListEqual(p.tolist(), p_t.tolist())
    
    def test_empty_category_without_ignore(self):
        """Test empty category case due to NA removal.
        """
        cat_data = np.array([[1,0,1,1,2]])
        cont_data = np.array([[-99,3,4,2,-99]])
        out_dict = napy.kruskal_wallis(cat_data, cont_data, nan_value=-99, ignore_empty_groups=False, use_numba=USE_NUMBA)
        s = out_dict['H']
        p = out_dict['p_unadjusted']
        self.assertTrue(np.isnan(s[0]))
        self.assertTrue(np.isnan(p[0]))
        
    def test_empty_category_with_ignore(self):
        """Test empty category case due to NA removal.
        """
        cat_data = np.array([[1,0,1,1,2]])
        cont_data = np.array([[-99,3,4,2,-99]])
        out_dict = napy.kruskal_wallis(cat_data, cont_data, nan_value=-99, ignore_empty_groups=True, use_numba=USE_NUMBA)
        s_napy = out_dict['H']
        p_napy = out_dict['p_unadjusted']
        s_sc, p_sc = sc.stats.kruskal([3], [4,2])
        self.assertAlmostEqual(s_napy[0], s_sc)
        self.assertAlmostEqual(p_napy[0], p_sc)
        
    def test_one_category(self):
        """Test case when only one category input category is given.
        """
        cat_data = np.array([[0,0,0,0,0]])
        cont_data = np.array([[2,3,3,1,3]])
        out_dict = napy.kruskal_wallis(cat_data, cont_data, use_numba=USE_NUMBA)
        p = out_dict['p_unadjusted']
        self.assertTrue(np.isnan(p[0]))
        
    def test_wrong_input(self):
        """Test if exception is raised when check_data parameter is used.
        """
        cont_data = np.random.rand(2,5)
        cat_data = np.array([[1,1,1,1,2], [0,2,1,0,0]])
        with self.assertRaises(ValueError):
            napy.kruskal_wallis(cat_data, cont_data, check_data=True, use_numba=USE_NUMBA)
        cat_data = np.array([[0,1,2,1,1], [3,0,1,1,1]])
        with self.assertRaises(ValueError):
            napy.kruskal_wallis(cat_data, cont_data, check_data=True, use_numba=USE_NUMBA)
    
    def test_threads(self):
        """Test on parallel correctness.
        """
        cat_data = np.array([[2,2,0,1], [1,1,0,1], [0,1,2,2]])
        cont_data = np.array([[2,3,3,4], [0,0,1,0], [3,3,0,1]])
        out_dict1 = napy.kruskal_wallis(cat_data, cont_data, threads=1, use_numba=USE_NUMBA)
        s1 = out_dict1['H']
        p1 = out_dict1['p_unadjusted']
        out_dict4 = napy.kruskal_wallis(cat_data, cont_data, threads=4, use_numba=USE_NUMBA)
        s4 = out_dict4['H']
        p4 = out_dict4['p_unadjusted']
        self.assertListEqual(s1.tolist(), s4.tolist())
        self.assertListEqual(p1.tolist(), p4.tolist())
        
    def test_basic_R(self):
        """Test basic functionality against R.
        """
        numpy2ri.activate()
        pandas2ri.activate()
        
        stats = importr('stats')
        cat_data = [1,0,2,1,2,0,0]
        cont_data = [4,1,1,0,5,2,1]
        data = pd.DataFrame({'continuous': cont_data, 'categorical': cat_data})
        napy_cat = np.array([[1,0,2,1,2,0,0]])
        napy_cont = np.array([[4,1,1,0,5,2,1]])

        # Convert the NumPy arrays to R vectors
        r_data = pandas2ri.py2rpy(data)
        cat_index = list(r_data.colnames).index('categorical')
        cat_col = robjects.vectors.FactorVector(r_data.rx2('categorical'))
        r_data[cat_index] = cat_col
        formula = Formula("continuous ~ categorical")

        # Call the R cor function
        kruskal_res = stats.kruskal_test(formula, data=r_data)
        
        r_statistic = kruskal_res.rx2(1)[0]
        p_value = kruskal_res.rx2(3)[0]

        out_dict = napy.kruskal_wallis(napy_cat, napy_cont, use_numba=USE_NUMBA)
        napy_corr = out_dict['H']
        napy_pvals = out_dict['p_unadjusted']
        self.assertAlmostEqual(r_statistic, napy_corr[0,0])
        self.assertAlmostEqual(r_statistic, napy_corr[0,0])
        self.assertAlmostEqual(p_value, napy_pvals[0,0])
        self.assertAlmostEqual(p_value, napy_pvals[0,0])

    def test_eta_squared(self):
        """
        Test calculation of eta squared effect size.
        """
        numpy2ri.activate()
        pandas2ri.activate()

        stats = importr('stats')
        rstatix = importr('rstatix')
        cat_data = [1, 0, 2, 1, 2, 0, 0]
        cont_data = [4, 1, 1, 0, 5, 2, 1]
        data = pd.DataFrame({'continuous': cont_data, 'categorical': cat_data})
        napy_cat = np.array([[1, 0, 2, 1, 2, 0, 0]])
        napy_cont = np.array([[4, 1, 1, 0, 5, 2, 1]])

        # Convert the NumPy arrays to R vectors
        r_data = pandas2ri.py2rpy(data)
        cat_index = list(r_data.colnames).index('categorical')
        cat_col = robjects.vectors.FactorVector(r_data.rx2('categorical'))
        r_data[cat_index] = cat_col
        formula = Formula("continuous ~ categorical")

        # Call the R cor function
        kruskal_res = rstatix.kruskal_effsize(formula, data=r_data)
        eta2 = kruskal_res.rx('effsize')[0][0]

        out_dict = napy.kruskal_wallis(napy_cat, napy_cont, use_numba=USE_NUMBA)
        eta_matrix = out_dict['eta2']
        self.assertAlmostEqual(eta2, eta_matrix[0,0])

class TestTTests(unittest.TestCase):
    def test_student_vs_scipy(self):
        """
        Test pairwise student's t-test against scipy implementation.
        """
        napy_bin = np.array([[1,0,1,1,0,0,0]])
        napy_cont = np.array([[2,4,1,1,3,2,3]])
        scipy_sample1 = [4,3,2,3]
        scipy_sample2 = [2,1,1]
        res = sc.stats.ttest_ind(scipy_sample1, scipy_sample2)
        sc_stat = res.statistic
        sc_pval = res.pvalue
        out_dict = napy.ttest(napy_bin, napy_cont, use_numba=USE_NUMBA)
        napy_stat = out_dict['t'][0][0]
        napy_pval = out_dict['p_unadjusted'][0][0]
        self.assertAlmostEqual(sc_stat, napy_stat)
        self.assertAlmostEqual(sc_pval, napy_pval)

    def test_basic_R(self):
        """Basic test case against R's t.test.
        """
        numpy2ri.activate()
        stats = importr('stats')

        # Convert the NumPy arrays to R vectors
        r_x = numpy2ri.py2rpy(np.array([4,3,2,3]))
        r_y = numpy2ri.py2rpy(np.array([2,1,1]))

        # Call the R ttest function
        test = stats.t_test(r_x, r_y, **{'var.equal': True})
        p_value = test.rx2('p.value')[0]
        test_statistic = test.rx2('statistic')[0]

        # Run napy's Student-t-test.
        bin_data = np.array([[1,0,1,1,0,0,0]])
        cont_data = np.array([[2,4,1,1,3,2,3]])
        out_dict = napy.ttest(bin_data, cont_data, use_numba=USE_NUMBA)
        napy_s = out_dict['t']
        napy_p = out_dict['p_unadjusted']
        self.assertAlmostEqual(test_statistic, napy_s[0][0])
        self.assertAlmostEqual(p_value, napy_p[0][0])

    def test_cohens_d_equal_vars(self):
        """
        Test calculation of Cohen's D in case of assumed equal variances.
        """
        lsr = importr('lsr')

        # Convert the NumPy arrays to R vectors
        arr1 = [4,3,2,3]
        arr2 = [2,1,1]
        arr1_r = robjects.FloatVector(arr1)
        arr2_r = robjects.FloatVector(arr2)

        # Call the R Cohen's D computation.
        cohens_d = lsr.cohensD(arr1_r, arr2_r)
        cohens_d = cohens_d[0]

        # Run napy's Student-t-test.
        bin_data = np.array([[1, 0, 1, 1, 0, 0, 0]])
        cont_data = np.array([[2, 4, 1, 1, 3, 2, 3]])
        out_dict = napy.ttest(bin_data, cont_data, use_numba=USE_NUMBA)
        napy_s = out_dict['cohens_d']

        self.assertAlmostEqual(cohens_d, napy_s[0][0])

    def test_welch_vs_scipy(self):
        """
        Test pairwise Welch's t-test against scipy implementation.
        """
        napy_bin = np.array([[1, 0, 1, 1, 0, 0, 0]])
        napy_cont = np.array([[2, 4, 1, 1, 3, 2, 3]])
        scipy_sample1 = [4, 3, 2, 3]
        scipy_sample2 = [2, 1, 1]
        res = sc.stats.ttest_ind(scipy_sample1, scipy_sample2, equal_var=False)
        sc_stat = res.statistic
        sc_pval = res.pvalue
        out_dict = napy.ttest(napy_bin, napy_cont, equal_var=False, use_numba=USE_NUMBA)
        napy_stat = out_dict['t'][0][0]
        napy_pval = out_dict['p_unadjusted'][0][0]
        self.assertAlmostEqual(sc_stat, napy_stat)
        self.assertAlmostEqual(sc_pval, napy_pval)

    def test_welch_vs_R(self):
        """
        Test Welch's test against R implementation.
        """
        numpy2ri.activate()
        stats = importr('stats')

        # Convert the NumPy arrays to R vectors
        r_x = numpy2ri.py2rpy(np.array([4, 3, 2, 3]))
        r_y = numpy2ri.py2rpy(np.array([2, 1, 1]))

        # Call the R ttest function
        test = stats.t_test(r_x, r_y, **{'var.equal': False})
        p_value = test.rx2('p.value')[0]
        test_statistic = test.rx2('statistic')[0]

        # Run napy's Student-t-test.
        bin_data = np.array([[1, 0, 1, 1, 0, 0, 0]])
        cont_data = np.array([[2, 4, 1, 1, 3, 2, 3]])
        out_dict = napy.ttest(bin_data, cont_data, equal_var=False, use_numba=USE_NUMBA)
        napy_s = out_dict['t']
        napy_p = out_dict['p_unadjusted']
        self.assertAlmostEqual(test_statistic, napy_s[0][0])
        self.assertAlmostEqual(p_value, napy_p[0][0])

    def test_welch_cohens_d(self):
        """
        Test Cohen's D calculation for Welch's t-test.
        """
        lsr = importr('lsr')

        # Convert the NumPy arrays to R vectors
        arr1 = [4, 3, 2, 3]
        arr2 = [2, 1, 1]
        arr1_r = robjects.FloatVector(arr1)
        arr2_r = robjects.FloatVector(arr2)

        # Call the R Cohen's D computation.
        cohens_d = lsr.cohensD(arr1_r, arr2_r, method='unequal')
        cohens_d = cohens_d[0]

        # Run napy's Student-t-test.
        bin_data = np.array([[1, 0, 1, 1, 0, 0, 0]])
        cont_data = np.array([[2, 4, 1, 1, 3, 2, 3]])
        out_dict = napy.ttest(bin_data, cont_data, equal_var=False, use_numba=USE_NUMBA)
        napy_s = out_dict['cohens_d']

        self.assertAlmostEqual(cohens_d, napy_s[0][0])

    def test_na_removal(self):
        """
        Test pairwise removal of NA values.
        """
        napy_bin = np.array([[1, 0, 1, 1, 0, 0, 0, -99]])
        napy_cont = np.array([[2, 4, 1, -99, 3, 2, 3, 4]])
        scipy_sample1 = [4, 3, 2, 3]
        scipy_sample2 = [2, 1]
        res = sc.stats.ttest_ind(scipy_sample1, scipy_sample2, equal_var=True)
        sc_stat = res.statistic
        sc_pval = res.pvalue
        out_dict = napy.ttest(napy_bin, napy_cont, nan_value=-99, equal_var=True, use_numba=USE_NUMBA)
        napy_stat = out_dict['t'][0][0]
        napy_pval = out_dict['p_unadjusted'][0][0]
        self.assertAlmostEqual(sc_stat, napy_stat)
        self.assertAlmostEqual(sc_pval, napy_pval)

    def test_axis_param(self):
        """Test functionality of axis input parameter.
        """
        bin_data = np.array([[1,0,0,1], [0,1,1,0], [1,1,0,0], [0,0,1,1]])
        cont_data = np.random.rand(4,4)
        out_dict = napy.ttest(bin_data, cont_data)
        s = out_dict['t']
        p = out_dict['p_unadjusted']
        cont_transp = cont_data.T.copy()
        bin_transp = bin_data.T.copy()
        out_dict_t = napy.ttest(bin_transp, cont_transp, axis=1, use_numba=USE_NUMBA)
        s_t = out_dict_t['t']
        p_t = out_dict['p_unadjusted']
        self.assertListEqual(s.tolist(), s_t.tolist())
        self.assertListEqual(p.tolist(), p_t.tolist())

    def test_singleton_category(self):
        """Test single-element category case due to NA removal.
        """
        bin_data = np.array([[0, 1, 1, 1, 0]])
        cont_data = np.array([[1, 3, 4, 2, -99]])
        out_dict = napy.ttest(bin_data, cont_data, nan_value=-99, use_numba=USE_NUMBA)
        s = out_dict['t']
        p = out_dict['p_unadjusted']
        self.assertTrue(np.isnan(s[0]))
        self.assertTrue(np.isnan(p[0]))

    def test_one_category(self):
        """Test case when only one category input category is given.
        """
        bin_data = np.array([[0, 0, 0, 0, 0]])
        cont_data = np.array([[2, 3, 3, 1, 3]])
        out_dict = napy.ttest(bin_data, cont_data, use_numba=USE_NUMBA)
        s = out_dict['t']
        p = out_dict['p_unadjusted']
        self.assertTrue(np.isnan(s[0]))
        self.assertTrue(np.isnan(p[0]))
        
    def test_parallel(self):
        """Test parallel functionality.
        """
        cat_data = np.array([[1,0,0,1], [0,1,1,0], [1,1,0,0], [0,0,1,1]])
        cont_data = np.random.rand(4,4)
        out_dict = napy.ttest(cat_data, cont_data, threads=1, use_numba=USE_NUMBA)
        s = out_dict['t']
        p = out_dict['p_unadjusted']
        out_dict_par = napy.ttest(cat_data, cont_data, threads=2, use_numba=USE_NUMBA)
        s_par = out_dict_par['t']
        p_par = out_dict_par['p_unadjusted']
        self.assertListEqual(s.tolist(), s_par.tolist())
        self.assertListEqual(p.tolist(), p_par.tolist())


class TestMultipleTestingCorrection(unittest.TestCase):
    def test_bonferroni_wo_diagonal(self):
        """
        Test basic pvalue correction with Bonferroni.
        """
        pvals = np.array([[0.0, 0.1, 0.3], [0.1, 0.0, 0.75], [0.4, 0.75, 0.0]])
        adj_pvals = napy._adjust_pvalues_bonferroni(pvals, ignore_diag=True)
        self.assertAlmostEqual(adj_pvals[0,1], 0.3)
        self.assertAlmostEqual(adj_pvals[0,2], 0.9)
        self.assertAlmostEqual(adj_pvals[2,1], 1.0)
        self.assertTrue(np.isnan(adj_pvals[0,0]))
        self.assertTrue(np.isnan(adj_pvals[1,1]))
        self.assertTrue(np.isnan(adj_pvals[2,2]))

    def test_bonferroni_with_diagonal(self):
        """
        Test basic pvalue correction with Bonferroni on non-squared pvalue matrix.
        """
        pvals = np.array([[0.1, 0.4, 0.15], [0.7, 0.05, 0.3]])
        adj_pvals = napy._adjust_pvalues_bonferroni(pvals, ignore_diag=False)
        self.assertAlmostEqual(adj_pvals[0,0], 0.6)
        self.assertAlmostEqual(adj_pvals[0,1], 1.0)
        self.assertAlmostEqual(adj_pvals[0,2], 0.9)
        self.assertAlmostEqual(adj_pvals[1,0], 1.0)
        self.assertAlmostEqual(adj_pvals[1,1], 0.3)
        self.assertAlmostEqual(adj_pvals[1,2], 1.0)

    def test_benjamini_hb_wo_diagonal(self):
        """
        Test basic Benajmini-Hochberg correction.
        """
        pvals = np.array([[0.0, 0.2, 0.1], [0.2, 0.0, 0.4], [0.1, 0.4, 0.0]])
        adj_pvals = napy._adjust_pvalues_fdr_control(pvals, method='bh', ignore_diag=True)
        self.assertAlmostEqual(adj_pvals[0,1], 0.3)
        self.assertAlmostEqual(adj_pvals[0,2], 0.3)
        self.assertAlmostEqual(adj_pvals[1,2], 0.4)
        self.assertAlmostEqual(adj_pvals[2,1], 0.4)
        self.assertAlmostEqual(adj_pvals[1,0], 0.3)
        self.assertAlmostEqual(adj_pvals[2,0], 0.3)
        self.assertTrue(np.isnan(adj_pvals[0,0]))
        self.assertTrue(np.isnan(adj_pvals[1,1]))
        self.assertTrue(np.isnan(adj_pvals[2,2]))

    def test_benjamini_hb_with_diagonal(self):
        """
        Test basic BH correction for non-squared pvalue matrix.
        """
        pvals = np.array([[0.2, 0.1, 0.5], [0.7, 0.3, 0.25]])
        adj_pvals = napy._adjust_pvalues_fdr_control(pvals, method='bh', ignore_diag=False)
        self.assertAlmostEqual(adj_pvals[0,0], 0.45)
        self.assertAlmostEqual(adj_pvals[0,1], 0.45)
        self.assertAlmostEqual(adj_pvals[0,2], 0.6)
        self.assertAlmostEqual(adj_pvals[1,0], 0.7)
        self.assertAlmostEqual(adj_pvals[1,1], 0.45)
        self.assertAlmostEqual(adj_pvals[1,2], 0.45)

class TestMWU(unittest.TestCase):
    def test_exact_against_scipy(self):
        """
        Test basic functionality of exact mode against scipy.
        """
        cat_napy = np.array([[0,0,0,0,1,1,1,1]])
        cont_napy = np.array([[3,5,1,2,4,6,7,8]])
        cat0 = [3,5,1,2]
        cat1 = [4,6,7,8]
        out_dict = napy.mwu(cat_napy, cont_napy, mode='exact', use_numba=USE_NUMBA)
        s = out_dict['U']
        p = out_dict['p_unadjusted']
        test = sc.stats.mannwhitneyu(cat0, cat1, method='exact')
        stat1 = test.statistic
        stat2 = 4*4 - stat1
        pval = test.pvalue
        self.assertAlmostEqual(p[0][0], pval)
        self.assertTrue(stat2 == s)

    def test_asymptotic_against_scipy(self):
        """
        Test basic functionality of asymptotic mode with ties against sicpy.
        """
        cat_napy = np.array([[0, 0, 0, 0, 1, 1, 1, 1]])
        cont_napy = np.array([[3, 5, 3, 2, 4, 6, 2, 8]])
        cat0 = [3, 5, 3, 2]
        cat1 = [4, 6, 2, 8]
        out_dict = napy.mwu(cat_napy, cont_napy, mode='asymptotic', use_numba=USE_NUMBA)
        s = out_dict['U']
        p = out_dict['p_unadjusted']
        test = sc.stats.mannwhitneyu(cat0, cat1, method='asymptotic', use_continuity=False)
        stat1 = test.statistic
        stat2 = 4 * 4 - stat1
        pval = test.pvalue
        self.assertAlmostEqual(p[0][0], pval)
        self.assertTrue(stat1 == s or stat2 == s)

    def test_auto_mode_with_ties(self):
        """
        Test auto mode with ties against scipy.
        """
        cat_napy = np.array([[0, 0, 0, 0, 1, 1, 1, 1]])
        cont_napy = np.array([[3, 5, 3, 2, 4, 6, 2, 8]])
        cat0 = [3, 5, 3, 2]
        cat1 = [4, 6, 2, 8]
        out_dict = napy.mwu(cat_napy, cont_napy, mode='auto', use_numba=USE_NUMBA)
        p = out_dict['p_unadjusted']
        test = sc.stats.mannwhitneyu(cat0, cat1, method='asymptotic', use_continuity=False)
        self.assertAlmostEqual(test.pvalue, p[0][0])

    def test_auto_mode_no_ties(self):
        """
        Test auto mode without ties against scipy.
        """
        cat_napy = np.array([[0, 0, 0, 0, 1, 1, 1, 1]])
        cont_napy = np.array([[3, 5, 1, 2, 4, 6, 7, 8]])
        cat0 = [3, 5, 1, 2]
        cat1 = [4, 6, 7, 8]
        out_dict = napy.mwu(cat_napy, cont_napy, mode='auto', use_numba=USE_NUMBA)
        p = out_dict['p_unadjusted']
        test = sc.stats.mannwhitneyu(cat0, cat1, method='exact', use_continuity=False)
        self.assertAlmostEqual(test.pvalue, p[0][0])

    def test_exact_vs_R(self):
        """
        Test exact Pvalue computation against R.
        """
        numpy2ri.activate()
        cat_data = np.array([[0,0,0,0,1,1,1,1]])
        cont_data = np.array([[1,3,9,4,2,6,7,12]])
        # Convert the NumPy arrays to R vectors
        r_x = numpy2ri.py2rpy(np.array([1,3,9,4]))
        r_y = numpy2ri.py2rpy(np.array([2,6,7,12]))

        # Call the R cor function
        wilcox = robjects.r['wilcox.test']
        res = wilcox(r_x, r_y, **{'exact': True, 'correct': False})
        pvalue_r = res.rx['p.value'][0][0]
        stat_r = res.rx['statistic'][0][0]
        stat_r_2 = 4*4 - stat_r

        # Call napy.
        out_dict = napy.mwu(cat_data, cont_data, mode='exact', use_numba=USE_NUMBA)
        s = out_dict['U']
        p = out_dict['p_unadjusted']
        self.assertTrue(s[0][0] == stat_r or s[0][0] == stat_r_2)
        self.assertAlmostEqual(p[0][0], pvalue_r)

    def test_asymptotic_vs_R(self):
        """
        Test asymptotic Pvalue calculation against R.
        """
        numpy2ri.activate()
        cat_data = np.array([[0, 0, 0, 0, 1, 1, 1, 1]])
        cont_data = np.array([[1, 3, 3, 4, 2, 1, 7, 12]])
        # Convert the NumPy arrays to R vectors
        r_x = numpy2ri.py2rpy(np.array([1, 3, 3, 4]))
        r_y = numpy2ri.py2rpy(np.array([2, 1, 7, 12]))

        # Call the R cor function
        wilcox = robjects.r['wilcox.test']
        res = wilcox(r_x, r_y, **{'exact': False, 'correct': False})
        pvalue_r = res.rx['p.value'][0][0]
        stat_r = res.rx['statistic'][0][0]
        stat_r_2 = 4*4 - stat_r
        # Call napy.
        out_dict = napy.mwu(cat_data, cont_data, mode='asymptotic', use_numba=USE_NUMBA)
        s = out_dict['U']
        p = out_dict['p_unadjusted']
        self.assertTrue(s[0][0] == stat_r or s[0][0] == stat_r_2)
        self.assertAlmostEqual(p[0][0], pvalue_r)

    def test_asymptotic_vs_R_no_ties(self):
        """
        Test asymptotic Pvalue calculation without ties.
        """
        numpy2ri.activate()
        cat_data = np.array([[0, 0, 0, 0, 1, 1, 1, 1]])
        cont_data = np.array([[1, 6, 3, 4, 2, 5, 7, 12]])
        # Convert the NumPy arrays to R vectors
        r_x = numpy2ri.py2rpy(np.array([1, 6, 3, 4]))
        r_y = numpy2ri.py2rpy(np.array([2, 5, 7, 12]))

        # Call the R cor function
        wilcox = robjects.r['wilcox.test']
        res = wilcox(r_x, r_y, **{'exact': False, 'correct': False})
        pvalue_r = res.rx['p.value'][0][0]
        stat_r = res.rx['statistic'][0][0]
        stat_r_2 = 4*4 - stat_r

        # Call napy.
        out_dict = napy.mwu(cat_data, cont_data, mode='asymptotic', use_numba=USE_NUMBA)
        s = out_dict['U']
        p = out_dict['p_unadjusted']
        self.assertTrue(s[0][0] == stat_r or s[0][0] == stat_r_2)
        self.assertAlmostEqual(p[0][0], pvalue_r)

    def test_effect_size_r_no_ties(self):
        """
        Test effect size computation against R in no tie case.
        """
        qnorm = robjects.r['qnorm']
        root = robjects.r['sqrt']
        wilcox = robjects.r['wilcox.test']

        # Compute effect size by hand from wilcox.test P-value.
        r_x = numpy2ri.py2rpy(np.array([4, 2, 5, 1]))
        r_y = numpy2ri.py2rpy(np.array([3, 6, 10, 8]))

        # Call the R test function.
        res = wilcox(r_x, r_y, **{'exact': False, 'correct': False})
        pvalue_r = res.rx['p.value'][0][0]
        z_value = qnorm(pvalue_r/2)
        z_value = z_value[0]
        eff_size_r = abs(z_value) / root(8)
        eff_size_r = eff_size_r[0]

        # Call napy effect size calculation.
        bin_data = np.array([[0,0,0,0,1,1,1,1]])
        cont_data = np.array([[4,2,5,1,3,6,10,8]])
        out_dict = napy.mwu(bin_data, cont_data, mode='exact', use_numba=USE_NUMBA)
        r = out_dict['r']
        self.assertAlmostEqual(abs(r[0][0]), eff_size_r)

    def test_effect_size_r_ties(self):
        """
        Test effect size computation in case with ties.
        """
        qnorm = robjects.r['qnorm']
        root = robjects.r['sqrt']
        wilcox = robjects.r['wilcox.test']

        # Compute effect size by hand from wilcox.test P-value.
        r_x = numpy2ri.py2rpy(np.array([4, 1, 5, 1]))
        r_y = numpy2ri.py2rpy(np.array([3, 6, 10, 5]))

        # Call the R test function.
        res = wilcox(r_x, r_y, **{'exact': False, 'correct': False})
        pvalue_r = res.rx['p.value'][0][0]
        z_value = qnorm(pvalue_r / 2)
        z_value = z_value[0]
        eff_size_r = abs(z_value) / root(8)
        eff_size_r = eff_size_r[0]

        # Call napy effect size calculation.
        bin_data = np.array([[0, 0, 0, 0, 1, 1, 1, 1]])
        cont_data = np.array([[4, 1, 5, 1, 3, 6, 10, 5]])
        out_dict = napy.mwu(bin_data, cont_data, mode='asymptotic', use_numba=USE_NUMBA)
        r = out_dict['r']
        self.assertAlmostEqual(abs(r[0][0]), eff_size_r)
        
    def test_rank_biserial_no_ties(self):
        """
        Test rank-biserial correlation against R effectsize package.
        """

        rank_biserial = robjects.r['rank_biserial']

        x = np.array([4, 2, 5, 1])
        y = np.array([3, 6, 10, 8])

        r_x = numpy2ri.py2rpy(x)
        r_y = numpy2ri.py2rpy(y)

        # R effectsize result
        res = rank_biserial(r_x, r_y)
        r_rb_r = float(res.rx2("r_rank_biserial")[0])

        # Python implementation
        bin_data = np.array([[0, 0, 0, 0, 1, 1, 1, 1]])
        cont_data = np.array([[4, 2, 5, 1, 3, 6, 10, 8]])

        out_dict = napy.mwu(
            bin_data,
            cont_data,
            mode="exact",
            use_numba=USE_NUMBA,
        )

        r_rb_py = out_dict["rb"][0][0]
        # Sign not relevant here, since this depends which group is chosen as reference.
        self.assertTrue(abs(r_rb_py) == abs(r_rb_r))
        
    def test_rank_biserial_with_ties(self):
        """
        Test rank-biserial correlation against R effectsize package.
        """

        rank_biserial = robjects.r['rank_biserial']

        x = np.array([4, 1, 5, 1])
        y = np.array([3, 6, 10, 8])

        r_x = numpy2ri.py2rpy(x)
        r_y = numpy2ri.py2rpy(y)

        # R effectsize result
        res = rank_biserial(r_x, r_y)
        r_rb_r = float(res.rx2("r_rank_biserial")[0])

        # Python implementation
        bin_data = np.array([[0, 0, 0, 0, 1, 1, 1, 1]])
        cont_data = np.array([[4, 1, 5, 1, 3, 6, 10, 8]])

        out_dict = napy.mwu(
            bin_data,
            cont_data,
            mode="exact",
            use_numba=USE_NUMBA,
        )

        r_rb_py = out_dict["rb"][0][0]
        # Sign not relevant here, since this depends which group is chosen as reference.
        self.assertTrue(abs(r_rb_py) == abs(r_rb_r))

    def test_na_removal(self):
        """
        Test basic NA removal functionality.
        """
        bin_data = np.array([[0,0,0,-99,  1,1,1,-99]])
        cont_data = np.array([[-99,2,3,3,  6,7,9,1]])
        out_dict = napy.mwu(bin_data, cont_data, mode='asymptotic', nan_value=-99, use_numba=USE_NUMBA)
        s = out_dict['U']
        p = out_dict['p_unadjusted']
        cat0 = [2,3]
        cat1 = [6,7,9]
        test = sc.stats.mannwhitneyu(cat0, cat1, use_continuity=False, method='asymptotic')
        stat_sc = test.statistic
        stat2 = 2*3 - stat_sc
        pval_sc = test.pvalue
        self.assertTrue(stat_sc == s[0][0] or stat2 == s[0][0])
        self.assertAlmostEqual(p[0][0], pval_sc)

    def test_empty_category(self):
        """
        Test edge case of emtpy category.
        """
        bin = np.array([[0,0,0,0,1]])
        cont = np.array([[1,2,3,4,-99]])
        out_dict = napy.mwu(bin, cont, nan_value=-99, use_numba=USE_NUMBA)
        s = out_dict['U']
        p = out_dict['p_unadjusted']
        self.assertTrue(np.isnan(s))
        self.assertTrue(np.isnan(p))
        
    def test_axis_param(self):
        """Test functionality of axis input parameter.
        """
        cat_data = np.array([[1,0,0,1], [0,1,1,0], [1,1,0,0], [0,0,1,1]])
        cont_data = np.random.rand(4,4)
        out_dict = napy.mwu(cat_data, cont_data)
        s = out_dict['U']
        p = out_dict['p_unadjusted']
        cont_transp = cont_data.T.copy()
        cat_transp = cat_data.T.copy()
        out_dic_t = napy.mwu(cat_transp, cont_transp, axis=1, use_numba=USE_NUMBA)
        s_t = out_dic_t['U']
        p_t = out_dic_t['p_unadjusted']
        self.assertListEqual(s.tolist(), s_t.tolist())
        self.assertListEqual(p.tolist(), p_t.tolist())
        
    def test_parallel(self):
        """Test parallel functionality.
        """
        cat_data = np.array([[1,0,0,1], [0,1,1,1], [1,1,0,0], [0,0,1,1]])
        cont_data = np.random.rand(4,4)
        out_dict = napy.mwu(cat_data, cont_data, threads=1, use_numba=USE_NUMBA)
        s = out_dict['U']
        p = out_dict['p_unadjusted']
        out_dict_par = napy.mwu(cat_data, cont_data, threads=2, use_numba=USE_NUMBA)
        s_par = out_dict_par['U']
        p_par = out_dict_par['p_unadjusted']
        self.assertListEqual(s.tolist(), s_par.tolist())
        self.assertListEqual(p.tolist(), p_par.tolist())

class TestPartialCorrelation(unittest.TestCase):
    
    def test_basic(self):
        """Test basic partial correlation computation against pingouin.
        """
        data = np.array([[2, 4, 15, 20, 25], [1, 2, 3, 4, 5], [0, 0, 1, 1, 1]])
        nan_value = -99
        out_dict = napy.partial_correlation(data, covar_indices=[2], nan_value=nan_value, axis=0, threads=1)
        napy_corr = out_dict['correlation']
        napy_pvalues = out_dict['p_unadjusted']
        pg_res = pg.pairwise_corr(data=pd.DataFrame(data.T), columns=[0, 1], covar=[2], method='pearson')
        pingouin_corr, pingouin_pvals = pg_res.loc[(pg_res['X'] == 0) & (pg_res['Y'] == 1), ['r', 'p_unc']].values[0]
        self.assertEqual(napy_corr[0,0], 1.0)
        self.assertEqual(napy_pvalues[0,0], 0.0)
        self.assertAlmostEqual(napy_corr[0,1], pingouin_corr)
        self.assertAlmostEqual(napy_pvalues[0,1], pingouin_pvals)
        self.assertAlmostEqual(napy_corr[0,1], napy_corr[1,0])
        self.assertAlmostEqual(napy_pvalues[0,1], napy_pvalues[1,0])
        
    def test_spearman(self):
        """Test spearman partial correlation computation against pingouin.
        """
        # data = np.array([[2, 4, 15, 20, 25], [1, 2, 3, 4, 5], [0, 0, 1, 1, 1]])
        
        data = np.array([[0.45014089, 0.44373781, 0.19470788, 0.98394716, 0.30484417,
        0.07198886, 0.872382  , 0.27466063, 0.2104761 , 0.95397527],
       [0.77095329, 0.61150653, 0.14472615, 0.74360492, 0.7405351 ,
        0.0914575 , 0.73462676, 0.53589392, 0.39264653, 0.23046924],
       [0.22026902, 0.5736392 , 0.34283945, 0.49448341, 0.98865515,
        0.90758143, 0.49381191, 0.71867761, 0.03594215, 0.04812974]])
        
        nan_value = -99
        out_dict = napy.partial_correlation(data, covar_indices=[2], method='spearman', nan_value=nan_value, axis=0, threads=1)
        napy_corr = out_dict['correlation']
        napy_pvals = out_dict['p_unadjusted']
        pg_res = pg.pairwise_corr(data=pd.DataFrame(data.T), columns=[0, 1], covar=[2], method='spearman')
        pingouin_corr, pingouin_pvals = pg_res.loc[(pg_res['X'] == 0) & (pg_res['Y'] == 1), ['r', 'p_unc']].values[0]
        self.assertEqual(napy_corr[0,0], 1.0)
        self.assertEqual(napy_pvals[0,0], 0.0)
        self.assertAlmostEqual(napy_corr[0,1], pingouin_corr)
        self.assertAlmostEqual(napy_pvals[0,1], pingouin_pvals)
        self.assertAlmostEqual(napy_corr[0,1], napy_corr[1,0])
        self.assertAlmostEqual(napy_pvals[0,1], napy_pvals[1,0])
        
    def test_no_confunder(self):
        """Test that partial correlation without confunder matches regular correlation.
        """
        data = np.array([[2, 4, 15, 20, 25], [1, 2, 3, 4, 5], [0, 0, 1, 1, 1]])
        nan_value = -99
        out_dict = napy.partial_correlation(data, covar_indices=[], nan_value=nan_value, axis=0, threads=1)
        napy_corr = out_dict['correlation']
        napy_pvalues = out_dict['p_unadjusted']
        scipy_corr, scipy_pvals = sc.stats.pearsonr(data[0], data[1])
        self.assertEqual(napy_corr[0,0], 1.0)
        self.assertEqual(napy_pvalues[0,0], 0.0)
        self.assertAlmostEqual(napy_corr[0,1], scipy_corr)
        self.assertAlmostEqual(napy_pvalues[0,1], scipy_pvals)
        self.assertAlmostEqual(napy_corr[0,1], napy_corr[1,0])
        self.assertAlmostEqual(napy_pvalues[0,1], napy_pvalues[1,0])
        
    def test_too_few_samples(self):
        """Test behavior when too few samples are present after NA removal.
        """
        data = np.array([[2, -99, 15, 20, 25], [1, 2, -99, 4, 5], [0, 0, 0, -99, 1]])
        nan_value = -99
        out_dict = napy.partial_correlation(data, covar_indices=[2], nan_value=nan_value, axis=0, threads=1)
        napy_corr = out_dict['correlation']
        napy_pvalues = out_dict['p_unadjusted']
        self.assertTrue(np.isnan(napy_corr[0,1]))
        self.assertTrue(np.isnan(napy_pvalues[0,1]))
        self.assertTrue(np.isnan(napy_corr[1,0]))
        self.assertTrue(np.isnan(napy_pvalues[1,0]))
        
    def test_large(self):
        """Test large data partial correlation computation against pingouin.
        """
        np.random.seed(0)
        data = np.random.rand(100, 20)
        nan_value = -99
        out_dict = napy.partial_correlation(data, covar_indices=list(range(10, 20)), nan_value=nan_value, axis=0, threads=1)
        napy_corr = out_dict['correlation']
        napy_pvals = out_dict['p_unadjusted']
        pg_res = pg.pairwise_corr(data=pd.DataFrame(data.T), columns=list(range(10)), covar=list(range(10, 20)), method='pearson')
        for i in range(10):
            for j in range(i+1, 10):
                pingouin_corr, pingouin_pvals = pg_res.loc[(pg_res['X'] == i) & (pg_res['Y'] == j), ['r', 'p_unc']].values[0]
                self.assertAlmostEqual(napy_corr[i,j], pingouin_corr)
                self.assertAlmostEqual(napy_pvals[i,j], pingouin_pvals)
                self.assertAlmostEqual(napy_corr[i,j], napy_corr[j,i])
                self.assertAlmostEqual(napy_pvals[i,j], napy_pvals[j,i])
                
    
    def test_multiple_confunder(self):
        """Test partial correltaion for multiple collumn as confunders
        """
        data = np.array([[2, 4, 6, 8, 25, 30, 35, 40, 45], [1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 0, 1, 1, 1, 1, 1], [5, 3, 2, 4, 6, 7, 5, 3, 6], [10, 12, 14, 11, 13, 15, 14, 16, 18]])
        nan_value = -99
        out_dict = napy.partial_correlation(data, covar_indices=[2, 3, 4], nan_value=nan_value, axis=0, threads=1)
        napy_corr = out_dict['correlation']
        napy_pvalues = out_dict['p_unadjusted']
        pg_res = pg.pairwise_corr(data=pd.DataFrame(data.T), columns=[0, 1], covar=[2, 3, 4], method='pearson')
        pingouin_corr, pingouin_pvals = pg_res.loc[(pg_res['X'] == 0) & (pg_res['Y'] == 1), ['r', 'p_unc']].values[0]
        self.assertEqual(napy_corr[0,0], 1.0)
        self.assertEqual(napy_pvalues[0,0], 0.0)
        self.assertAlmostEqual(napy_corr[0,1], pingouin_corr)
        self.assertAlmostEqual(napy_pvalues[0,1], pingouin_pvals)
        self.assertAlmostEqual(napy_corr[0,1], napy_corr[1,0])
        self.assertAlmostEqual(napy_pvalues[0,1], napy_pvalues[1,0])
        
    def test_confunder_variable_overlap(self):
        """ Test behaviour of partial correlation when confunder variables overlap with target variables.
        """
        data = np.array([[2, 4, 15, 20, 25], [1, 2, 3, 4, 5], [0, 0, 1, 1, 1]])
        nan_value = -99
        out_dict = napy.partial_correlation(data, covar_indices=[1], nan_value=nan_value, axis=0, threads=1)
        napy_corr = out_dict['correlation']
        napy_pvalues = out_dict['p_unadjusted']
        self.assertTrue(np.isnan(napy_corr[0,1]))
        self.assertTrue(np.isnan(napy_pvalues[0,1]))
        self.assertTrue(np.isnan(napy_corr[1,0]))
        self.assertTrue(np.isnan(napy_pvalues[1,0]))
    
    def test_nans(self):
        """Test partial correlation computation with NaNs against pingouin.
        """
        nan_value = -99
        data = np.array([[2, 4, 15, 20, nan_value, 30, 35], [1, 2, 3, 4, 5, 6, 7], [0, 0, 1, 1, 1, nan_value, 1]])
        out_dict = napy.partial_correlation(data, covar_indices=[2], nan_value=nan_value, axis=0, threads=1)
        napy_corr = out_dict['correlation']
        napy_pvals = out_dict['p_unadjusted']
        data = pd.DataFrame(data.T)
        data[data == nan_value] = pd.NA
        pg_res = pg.pairwise_corr(data=data, columns=[0, 1], covar=[2], method='pearson', nan_policy='pairwise')
        pingouin_corr, pingouin_pvals = pg_res.loc[(pg_res['X'] == 0) & (pg_res['Y'] == 1), ['r', 'p_unc']].values[0]
        self.assertEqual(napy_corr[0,0], 1.0)
        self.assertEqual(napy_pvals[0,0], 0.0)
        self.assertAlmostEqual(napy_corr[0,1], pingouin_corr)
        self.assertAlmostEqual(napy_pvals[0,1], pingouin_pvals)
        self.assertAlmostEqual(napy_corr[0,1], napy_corr[1,0])
        self.assertAlmostEqual(napy_pvals[0,1], napy_pvals[1,0])
        
    def test_axis(self):
        """Test axis parameter functionality.
        """
        data = np.array([[2, 4, 15, 20, 25], [1, 2, 3, 4, 5], [0, 0, 1, 1, 1]])
        nan_value = -99
        out_dict = napy.partial_correlation(data, covar_indices=[2], nan_value=nan_value, axis=0, threads=1)
        napy_corr = out_dict['correlation']
        napy_pvals = out_dict['p_unadjusted']
        data_t = data.T.copy()
        out_dict_t = napy.partial_correlation(data_t, covar_indices=[2], nan_value=nan_value, axis=1, threads=1)
        napy_corr_t = out_dict_t['correlation']
        napy_pvals_t = out_dict_t['p_unadjusted']
        np.testing.assert_equal(napy_corr.tolist(), napy_corr_t.tolist())
        np.testing.assert_equal(napy_pvals.tolist(), napy_pvals_t.tolist())
        
    def test_parallel(self):
        """Test parallel functionality.
        """
        data = np.array([[2, 4, 15, 20, 25], [1, 2, 3, 4, 5], [0, 0, 1, 1, 1]])
        nan_value = -99
        out_dict = napy.partial_correlation(data, covar_indices=[2], nan_value=nan_value, axis=0, threads=1)
        napy_corr = out_dict['correlation']
        napy_pvals = out_dict['p_unadjusted']
        out_dict_par = napy.partial_correlation(data, covar_indices=[2], nan_value=nan_value, axis=0, threads=4)
        napy_corr_par = out_dict_par['correlation']
        napy_pvals_par = out_dict_par['p_unadjusted']
        np.testing.assert_equal(napy_corr.tolist(), napy_corr_par.tolist())
        np.testing.assert_equal(napy_pvals.tolist(), napy_pvals_par.tolist())


if __name__ == "__main__":
    robjects.r['options'](warn=-1)
    unittest.main(verbosity=2)

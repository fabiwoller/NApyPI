from .wrapper import pearsonr, spearmanr, chi_squared, anova
from .wrapper import kruskal_wallis, ttest, mwu, _adjust_pvalues_bonferroni, _adjust_pvalues_fdr_control
from importlib.metadata import version, PackageNotFoundError

__all__ = ["pearsonr",
           "spearmanr",
           "chi_squared",
           "anova",
           "kruskal_wallis",
           "ttest",
           "mwu", 
           "_adjust_pvalues_bonferroni",
           "_adjust_pvalues_fdr_control"]

try:
    __version__ = version("napypi")
except PackageNotFoundError:
    __version__ = "unknown"
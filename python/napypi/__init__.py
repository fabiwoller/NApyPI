from .wrapper import pearsonr, spearmanr, chi_squared, anova
from .wrapper import kruskal_wallis, ttest, mwu
from importlib.metadata import version, PackageNotFoundError

__all__ = ["pearsonr",
           "spearmanr",
           "chi_squared",
           "anova",
           "kruskal_wallis",
           "ttest",
           "mwu"]

try:
    __version__ = version("napypi")
except PackageNotFoundError:
    __version__ = "unknown"
"""
ABC Class of Analysis Report
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

from ..analyzer.analyze_result import AnalyzeResult

__all__ = [
    '_Report'
]


class _Report(ABC):
    """
    ABC Class of Analysis Report
    """
    @abstractmethod
    def __init__(self, **kwargs: Dict[str, Any]):
        """
        Parameters
        ----------
        kwargs: Dict[str, Any]
            params for each Report
        """
        pass

    @abstractmethod
    def get_report(self, analyze_result: AnalyzeResult) -> Any:
        """
        Parameters
        ----------
        analyze_result:
            target result of statistical analysis.
            Report object will generate analysis report from this data.

        Return
        ------
        report: Any
            Analysis report in some format

        """
        pass

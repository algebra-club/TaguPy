"""
ABC Class of Analysis Reporter
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Type


AnalysisResult = Type[NamedTuple]


class _Reporter(ABC):
    """
    ABC Class of Analysis Reporter

    Methods
    -------
    get_report(analyze_result: AnalyzeResult) -> Any
    """

    @abstractmethod
    def __init__(self, **kwargs: Dict[str, Any]):
        """
        Parameters
        ----------
        kwargs: Dict[str, Any]
            params for each Reporter
        """
        pass

    @abstractmethod
    def get_report(self, analyze_result: AnalysisResult) -> Any:
        """
        Generate Report in some format

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

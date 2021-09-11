import numpy as np
import pandas as pd
from pandas import Series
from typing import NamedTuple
from tagupy.type import _Analyzer as Analyzer


class EDAResult(NamedTuple):
    head: Series.head
    tail: Series.tail
    max: Series.max
    min: Series.min
    mean: Series.mean
    median: Series.median
    var: Series.var
    corr: Series.corr
    quantile: Series.quantile
    isna: Series.isna
    count: Series.count
    skew: Series.skew
    kurt: Series.kurt


class EDA(Analyzer):
    """
    """

    def __init__(self, dataMatrix: np.ndarray):
        if not isinstance(dataMatrix, np.ndarray):
            raise TypeError(
                f'argument dataMatrix expected numpy.ndarray, \
                    got {type(dataMatrix)}::{dataMatrix}'
            )

        self.dataframe = pd.DataFrame(dataMatrix)

    @property
    def head(self):
        return self.dataframe.head

    @property
    def tail(self):
        return self.dataframe.tail

    @property
    def max(self):
        return self.dataframe.max

    @property
    def min(self):
        return self.dataframe.min

    @property
    def mean(self):
        return self.dataframe.mean

    @property
    def median(self):
        return self.dataframe.median

    @property
    def var(self):
        return self.dataframe.var

    @property
    def corr(self):
        return self.dataframe.corr

    @property
    def quantile(self):
        return self.dataframe.quantile

    @property
    def isna(self):
        return self.dataframe.isna

    @property
    def count(self):
        return self.dataframe.count

    @property
    def skew(self):
        return self.dataframe.skew

    @property
    def kurt(self):
        return self.dataframe.kurt

    def analyze(self):
        return EDAResult(
            head=self.head,
            tail=self.tail,
            max=self.max,
            min=self.min,
            mean=self.mean,
            median=self.median,
            var=self.var,
            corr=self.corr,
            quantile=self.quantile,
            isna=self.isna,
            count=self.count,
            skew=self.skew,
            kurt=self.kurt,
        )

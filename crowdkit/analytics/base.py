from typing import Optional, Union

import attr
import pandas as pd




@attr.s
class BaseCrowdMetricsCalculator:
    """ This is a base class for all crowd metrics aggregators"""

    def calculate(self, answers: pd.DataFrame) -> Union[float, pd.Series]:
        raise NotImplementedError()


@attr.s
class BaseTwoDatasetsCrowdMetricsCalculator:
    """ This is a base class for all crowd metrics aggregators"""

    def calculate(self, dataset1: pd.DataFrame, dataset2: Optional[pd.DataFrame] = None) -> Union[float, pd.Series]:
        raise NotImplementedError()


def check_answers(answers: pd.DataFrame) -> None:
    if not isinstance(answers, pd.DataFrame):
        raise TypeError('Working only with pandas DataFrame')
    assert 'task' in answers, 'There is no "task" column in answers'
    assert 'worker' in answers, 'There is no "worker" column in answers'
    assert 'label' in answers, 'There is no "label" column in answers'

class ComputeBy(str, Enum):
    TASK = 'task'
    WORKER = 'worker'
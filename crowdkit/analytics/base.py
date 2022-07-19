from typing import Union

import attr
import pandas as pd




@attr.s
class BaseCrowdMetricsCalculator:
    """ This is a base class for all crowd metrics aggregators"""

    def calculate(self, answers: pd.DataFrame) -> Union[float, pd.Series]:
        raise NotImplementedError()


def _check_answers(answers: pd.DataFrame) -> None:
    if not isinstance(answers, pd.DataFrame):
        raise TypeError('Working only with pandas DataFrame')
    assert 'task' in answers, 'There is no "task" column in answers'
    assert 'worker' in answers, 'There is no "worker" column in answers'
    assert 'label' in answers, 'There is no "label" column in answers'
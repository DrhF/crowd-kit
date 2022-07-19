from typing import Any, Union

import pandas as pd

from crowdkit.aggregation import MajorityVote
from crowdkit.aggregation.base import BaseClassificationAggregator
from crowdkit.analytics.base import _check_answers, BaseCrowdMetricsCalculator


class ConsistencyCalculator(BaseCrowdMetricsCalculator):
    """
        Consistency metric: posterior probability of aggregated label given workers skills
        calculated using standard Majority Vote model.

        Args:
            aggregator (aggregation.base.BaseClassificationAggregator): aggregation method, default: MajorityVote
            by_task (bool): if set, returns consistencies for every task in provided series.

        Returns:
            Union[float, pd.Series]
    """

    def __init__(self, aggregator: BaseClassificationAggregator = MajorityVote(), by_task: bool = False):
        self.aggregator = aggregator
        self.by_task = by_task

    @staticmethod
    def __task_consistency(row: pd.Series) -> float:
        """Posterior probability for a single task"""
        return row[row['aggregated_label']] / row['denominator'] if row['denominator'] != 0 else 0.0

    @staticmethod
    def __label_probability(row: pd.Series, label: Any, n_labels: int) -> float:
        """Numerator in the Bayes formula"""
        return row['skill'] if row['label'] == label else (1.0 - row['skill']) / (n_labels - 1)

    def calculate(self, answers: pd.DataFrame) -> Union[float, pd.Series]:
        """
            Calculates consistency metrics

            Args:
                answers: A data frame containing `task`, `worker` and `label` columns.

            Returns:
                Series if by_task is True, float value otherwise.
        """
        _check_answers(answers)
        aggregated = self.aggregator.fit_predict(answers)
        if workers_skills is None:
            if hasattr(self.aggregator, 'skills_'):
                workers_skills = self.aggregator.skills_  # type: ignore
            else:
                raise AssertionError('This aggregator is not supported. Please, provide workers skills.')

        answers = answers.copy(deep=False)
        answers.set_index('task', inplace=True)
        answers = answers.reset_index().set_index('worker')
        answers['skill'] = workers_skills
        answers.reset_index(inplace=True)

        labels = pd.unique(answers.label)
        for label in labels:
            answers[label] = answers.apply(lambda row: self._label_probability(row, label, len(labels)), axis=1)

        labels_proba = answers.groupby('task').prod()
        labels_proba['aggregated_label'] = aggregated
        labels_proba['denominator'] = labels_proba[list(labels)].sum(axis=1)

        consistencies = labels_proba.apply(self._task_consistency, axis=1)

        if self.by_task:
            return consistencies
        else:
            return consistencies.mean()

from typing import Any, Union, Optional

import pandas as pd
from enum import Enum
from scipy.stats import entropy

from crowdkit.aggregation import MajorityVote
from crowdkit.aggregation.base import BaseClassificationAggregator
from crowdkit.analytics.base import _check_answers, BaseCrowdMetricsCalculator


class ComputeBy(str, Enum):
    TASK = 'task'
    WORKER = 'worker'


class UncertaintyCalculator(BaseCrowdMetricsCalculator):
    """
        Label uncertainty metric: entropy of labels probability distribution.
        Computed as Shannon's Entropy with label probabilities computed either for tasks or workers:
        $$H(L) = -\sum_{label_i \in L} p(label_i) \cdot \log(p(label_i))$$
        Args:
            aggregator (aggregation.base.BaseClassificationAggregator): aggregation method, default: MajorityVote
            compute_by: what to compute uncertainty for. If 'task', compute uncertainty of answers per task.
                If 'worker', compute uncertainty for each worker.
            aggregate: If true, return the mean uncertainty, otherwise return uncertainties for each task or worker.
    """

    def __init__(self, compute_by: ComputeBy = 'task', aggregate: bool = True,
                 aggregator: Optional[BaseClassificationAggregator] = None):
        self.compute_by = compute_by
        self.aggregate = aggregate
        self.aggregator = aggregator

    @staticmethod
    def __label_probability(row: pd.Series, label: Any, n_labels: int) -> float:
        """Numerator in the Bayes formula"""
        return row['skill'] if row['label'] == label else (1.0 - row['skill']) / (n_labels - 1)

    def calculate(self, answers: pd.DataFrame) -> Union[float, pd.Series]:
        """
            Calculates uncertainty metrics

            Args:
                answers: A data frame containing `task`, `worker` and `label` columns.

            Returns:
                Series if aggregate is True, float value otherwise.
        """
        _check_answers(answers)
        if self.aggregator is not None:
            self.aggregator.fit(answers)
            if hasattr(self.aggregator, 'skills_'):
                workers_skills = self.aggregator.skills_
            else:
                raise AssertionError('This aggregator is not supported. Please, provide workers skills.')
        else:
            workers_skills = None

        answers = answers.copy(deep=False)
        answers = answers.set_index('worker')
        answers['skill'] = workers_skills if workers_skills is not None else 1
        if answers['skill'].isnull().any():
            missing_workers = set(answers[answers.skill.isnull()].index.tolist())
            raise AssertionError(f'Did not provide skills for workers: {missing_workers}.'
                                 f'Please provide workers skills.')
        answers.reset_index(inplace=True)
        labels = pd.unique(answers.label)
        for label in labels:
            answers[label] = answers.apply(lambda row: self.__label_probability(row, label, len(labels)), axis=1)

        labels_proba = answers.groupby(self.compute_by).sum()
        uncertainties = labels_proba.apply(lambda row: entropy(row[labels] / (sum(row[labels]) + 1e-6)), axis=1)
        if self.aggregate:
            return uncertainties.mean()
        else:
            return uncertainties
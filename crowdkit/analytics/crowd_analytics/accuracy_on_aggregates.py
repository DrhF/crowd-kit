import pandas as pd

from typing import Optional, Union
from crowdkit.aggregation.base import BaseClassificationAggregator
from crowdkit.aggregation.classification.majority_vote import MajorityVote
from crowdkit.analytics.base import BaseTwoDatasetsCrowdMetricsCalculator, ComputeBy, check_answers


class AccuracyOnAggregatesCalculator(BaseTwoDatasetsCrowdMetricsCalculator):
    """
        Accuracy on aggregates: a fraction of worker's answers that match the aggregated one.

    Args:
        aggregator: aggregation algorithm. default: MajorityVote
        by: if set, returns accuracies for every worker in provided data frame. Otherwise,
            returns an average accuracy of all workers.
    """

    def __init__(self, aggregator: Optional[BaseClassificationAggregator] = MajorityVote(),
                 by: ComputeBy = ComputeBy.TASK):
        self.aggregator = aggregator
        self.by = by


    def __get_accuracy(self, data: pd.DataFrame, true_labels: pd.Series) -> pd.Series:
        """Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
            true_labels (Series): Tasks' ground truth labels.
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's ground truth label.

        Returns:
            Series: workers' skills.
                A pandas.Series index by workers and holding corresponding worker's skill
        """
        if 'weight' in data.columns:
            data = data[['task', 'worker', 'label', 'weight']]
        else:
            data = data[['task', 'worker', 'label']]

        if data.empty:
            data['true_label'] = []
        else:
            data = data.join(pd.Series(true_labels, name='true_label'), on='task')

        data = data[data.true_label.notna()]

        if 'weight' not in data.columns:
            data['weight'] = 1
        data.eval('score = weight * (label == true_label)', inplace=True)

        data = data.sort_values('score').drop_duplicates(['task', 'worker', 'label'], keep='last')

        if self.by == ComputeBy.WORKER:
            data = data.groupby(self.by)

        return data.score.sum() / data.weight.sum()
    
    
    def calculate(self, dataset1: pd.DataFrame, dataset2: Optional[pd.DataFrame] = None) -> Union[float, pd.Series]:
        """
            Calculates uncertainty metrics

            Args:
                dataset1: answers, a data frame containing `task`, `worker` and `label` columns.
                dataset2: Optional, aggregates: aggregated answers for provided tasks. A data frame containing `task`, `worker` and `label` columns.

            Returns:
                Series if aggregate is True, float value otherwise.
        """
        answers, aggregates = dataset1, dataset2
        if aggregates is None:
            aggregates = aggregator.fit_predict(answers)  # type: ignore
        
        check_answers(answers)
        
        return self.__get_accuracy(answers, aggregates)
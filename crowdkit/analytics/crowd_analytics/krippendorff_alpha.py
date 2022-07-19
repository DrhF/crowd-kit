from typing import Callable, Hashable, List, Tuple, Any, Union
import pandas as pd
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics.distance import binary_distance

from crowdkit.analytics.base import BaseCrowdMetricsCalculator, check_answers


class KrippendorffAlphaCalculator(BaseCrowdMetricsCalculator):
    """Inter-annotator agreement coefficient (Krippendorff 1980).

        Amount that annotators agreed on label assignments beyond what is expected by chance.
        The value of alpha should be interpreted as follows.
            alpha >= 0.8 indicates a reliable annotation,
            alpha >= 0.667 allows making tentative conclusions only,
            while the lower values suggest the unreliable annotation.

        Args:
            answers: A data frame containing `task`, `worker` and `label` columns.
            distance: Distance metric, that takes two arguments,
                and returns a value between 0.0 and 1.0
                By default: binary_distance (0.0 for equal labels 1.0 otherwise).

        Returns:
            Float value.

        Examples:
            Consistent answers.

            >>> alpha_krippendorff(pd.DataFrame.from_records([
            >>>     {'task': 'X', 'worker': 'A', 'label': 'Yes'},
            >>>     {'task': 'X', 'worker': 'B', 'label': 'Yes'},
            >>>     {'task': 'Y', 'worker': 'A', 'label': 'No'},
            >>>     {'task': 'Y', 'worker': 'B', 'label': 'No'},
            >>> ]))
            1.0

            Partially inconsistent answers.

            >>> alpha_krippendorff(pd.DataFrame.from_records([
            >>>     {'task': 'X', 'worker': 'A', 'label': 'Yes'},
            >>>     {'task': 'X', 'worker': 'B', 'label': 'Yes'},
            >>>     {'task': 'Y', 'worker': 'A', 'label': 'No'},
            >>>     {'task': 'Y', 'worker': 'B', 'label': 'No'},
            >>>     {'task': 'Z', 'worker': 'A', 'label': 'Yes'},
            >>>     {'task': 'Z', 'worker': 'B', 'label': 'No'},
            >>> ]))
            0.4444444444444444
        """

    def __init__(self, distance_callable: Callable[[Hashable, Hashable], float] = binary_distance):
        self.distance_callable = distance_callable

    def calculate(self, answers: pd.DataFrame) -> Union[float, pd.Series]:
        """
            Calculates Krippendorff's alpha metrics

            Args:
                answers: A data frame containing `task`, `worker` and `label` columns.

            Returns:
                Float value.
        """
        check_answers(answers)
        data: List[Tuple[Any, Hashable, Hashable]] = answers[['worker', 'task', 'label']].values.tolist()
        return AnnotationTask(data, self.distance_callable).alpha()
import abc
from typing import Dict, Union

from allenact.utils.experiment_utils import ScalarMeanTracker
from allenact.utils.system import get_logger


class EngineSensor(abc.ABC):
    """Abstract class for class which computes metrics using training and inference
    engine."""

    @abc.abstractmethod
    def __call__(self, engine, **kwargs) -> dict:
        """Returns `metrics` from engine.

        # Parameters

        engine: Training or inference engine.
        """
        raise NotImplementedError


class MetricsEngineSensor(abc.ABC):
    """Engine sensor to fetch training or validation metrics at some interval."""

    aggregated_task_metrics: ScalarMeanTracker = ScalarMeanTracker()
    reset_interval: int = 100000
    last_log: int = 0
    reset_on_last_step: int = False

    def __init__(self, reset_interval: int = 100000):
        self.reset_interval = reset_interval

    @staticmethod
    def _metrics_dict_is_empty(
        single_task_metrics_dict: Dict[str, Union[float, int]]
    ) -> bool:
        return (
            len(single_task_metrics_dict) == 0
            or (
                len(single_task_metrics_dict) == 1
                and "task_info" in single_task_metrics_dict
            )
            or (
                "success" in single_task_metrics_dict
                and single_task_metrics_dict["success"] is None
            )
        )

    def __call__(self, engine, **kwargs) -> dict:
        """Returns `metrics` from engine.

        # Parameters

        engine: Training or inference engine.

        # Returns

        A dict with:

        metrics: current training or validation metrics
        """
        info = {"metrics": {}}
        # if self.reset_on_last_step:
        #     get_logger().info(
        #         "\n\nAggregated metrics after reset: {}, {}".format(
        #             self.aggregated_task_metrics.means(),
        #             [(x["success"], x["spl"]) for x in engine.single_process_metrics],
        #         )
        #     )

        single_process_metrics = engine.single_process_metrics
        for single_task_metrics_dict in single_process_metrics:
            if self._metrics_dict_is_empty(single_task_metrics_dict):
                continue

            self.aggregated_task_metrics.add_scalars(
                {k: v for k, v in single_task_metrics_dict.items() if k != "task_info"}
            )

        info["metrics"] = self.aggregated_task_metrics.means()
        if self.reset_on_last_step:
            # get_logger().info(
            #     "\n\nAggregated metrics after reset: {}".format(
            #         self.aggregated_task_metrics.means()
            #     )
            # )
            self.reset_on_last_step = False
        return info

    def means(self) -> Dict[str, float]:
        return self.aggregated_task_metrics.means()

    def reset(self, total_steps: int) -> None:
        if total_steps - self.last_log >= self.reset_interval:
            get_logger().info(
                "\n\nAggregated metrics before reset: {}".format(
                    self.aggregated_task_metrics.means()
                )
            )
            self.aggregated_task_metrics.reset()
            self.last_log = total_steps
            self.reset_on_last_step = True
            get_logger().info(
                "\n\nResetting curriculum metrics at: {}".format(total_steps)
            )

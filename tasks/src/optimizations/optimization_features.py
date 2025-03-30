"""Techniques for selection best features by models """

from src.optimizations.optimization_enum import (
    FileProcessForwardBackward, 
    ProcedureForwardBackward,
    ActionProcedure,
    )


class FeatureForwardBackwardSelection:
    """Run the backward feature selection and whe it finish
        run the forward feature selection
    """

    def __init__(
        self,
        file_process: FileProcessForwardBackward,
    ) -> None:
        self._file_process = file_process

    def run(self) -> None:
        """Run the backward feature selection and whe it finish run
        the forward feature selection
        """
        features = self._file_process.initial_features
        for step in self._file_process.get_steps():
            # step_action = (
            dd = ProcedureForwardBackward()
            if step.action == ActionProcedure.BACKWARD:
                features = self._backward_feature_selection(
                    intial_features=features,
                    all_features=features
                )
   

    def _backward_feature_selection(self) -> None:
        """Run the backward feature selection
        """
        pass

    def _forward_feature_selection(self) -> None:
        """Run the forward feature selection
        """
        pass



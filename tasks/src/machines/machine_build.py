"""Create some machine
    """

from src.machines.machine_enums import MachineNames
from src.machines.machine import (
    MachineRegression, BayesianRegression, ExtremeGradientBoostRegression)


def machine_build_regression_optimization(
        machine_name: MachineNames,
        deep_decimal: int,
) -> MachineRegression:
    """Build some machine for optimization with random values

    Args:
        machine_definition (MachineJson): Machine definition

    Raises:
        NotImplementedError: The machine don't exist

    Returns:
        MachineJson: Return child of Machine Json to run it
    """
    # if machine_definition.name_machine == MachineNames.RANDOM_FOREST_REGRESSION:
    #     return RandomForestRegression(
    #         n_estimators=machine_definition.n_estimators,
    #         random_sate=machine_definition.random_state,
    #         n_jobs=machine_definition.n_jobs,
    #     )
    if machine_name == MachineNames.EXTREME_GRADIENT_BOOSTING_REGRESSION:
        machine = ExtremeGradientBoostRegression()
        machine.force_mutate_hyper_parameters(deep_decimal=deep_decimal)
        return machine
    if machine_name == MachineNames.BAYESIAN_REGRESSION:
        machine = BayesianRegression()
        machine.force_mutate_hyper_parameters(deep_decimal=deep_decimal)
        return machine
    # if machine_definition.name_machine == MachineNames.LASSO_REGRESSION:
    #     return LassoRegression()
    # if machine_definition.name_machine == MachineNames.SUPPORT_VECTOR_REGRESSION:
    #     return SupportVectorRegression()
    raise NotImplementedError("Don't exist machine to run it")

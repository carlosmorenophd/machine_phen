"""Create some machine
    """

from src.machines.machine_enums import MachineNames, MachineJson
from src.machines.machine import MachineRegression, BayesianRegression


def machine_build_regression_optimization(machine_definition: MachineJson) -> MachineRegression:
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
    # if machine_definition.name_machine == MachineNames.EXTREME_GRADIENT_BOOSTING_REGRESSION:
    #     return ExtremeGradientBoostRegression()
    if machine_definition.name_machine == MachineNames.BAYESIAN_REGRESSION:
        machine = BayesianRegression()
        return machine
    # if machine_definition.name_machine == MachineNames.LASSO_REGRESSION:
    #     return LassoRegression()
    # if machine_definition.name_machine == MachineNames.SUPPORT_VECTOR_REGRESSION:
    #     return SupportVectorRegression()
    raise NotImplementedError("Don't exist machine to run it")

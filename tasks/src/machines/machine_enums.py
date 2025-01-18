"""Skelton for class to predict"""
from dataclasses import dataclass
from enum import Enum

from numpy import ndarray
from src.helpers.file_access import FolderCache


class SupportVectorKernelEnum(Enum):
    """Kernel for SVR

    Args:
        Enum (_type_): _description_
    """
    LINEAR = 'linear'
    POLY = 'poly'
    PRE_COMPUTER = 'precomputed'
    RBF = 'rbf'
    SIGMOID = 'sigmoid'


def cast_kernel_svm(input_kernel: str):
    """Function to cast text to support vector regression kernel enum

    Args:
        input (str): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if input_kernel == SupportVectorKernelEnum.LINEAR.value:
        return SupportVectorKernelEnum.LINEAR
    if input_kernel == SupportVectorKernelEnum.POLY.value:
        return SupportVectorKernelEnum.POLY
    if input_kernel == SupportVectorKernelEnum.PRE_COMPUTER.value:
        return SupportVectorKernelEnum.PRE_COMPUTER
    if input_kernel == SupportVectorKernelEnum.RBF.value:
        return SupportVectorKernelEnum.RBF
    if input_kernel == SupportVectorKernelEnum.SIGMOID.value:
        return SupportVectorKernelEnum.SIGMOID
    raise ValueError(f"Not kernel valid {input_kernel}")


class MachineNames(Enum):
    """Enum for machine names"""
    RANDOM_FOREST_REGRESSION = "random_forest_regression"
    EXTREME_GRADIENT_BOOSTING_REGRESSION = "extreme_gradient_boost_regression"
    BAYESIAN_REGRESSION = "bayesian_prediction_regression"
    LASSO_REGRESSION = "lasso_regression"
    SUPPORT_VECTOR_REGRESSION = "support_vector_regression"


class MachinesTypes(Enum):
    """Enum for machine types"""
    REGRESSION = "regression"
    CLASSIFICATION = "clarification"


class MachineJson:
    """Generic class for Machines Json
    """

    def __init__(self, name_machine: MachineNames, type_machine: MachinesTypes) -> None:
        self.name_machine = name_machine
        self.type_machine = type_machine

    def __str__(self) -> str:
        return f"machine_name: {
            self.name_machine.value
        }"


@dataclass
class FileAccessRunnerProperties:
    """Minimal parameter to load, split  the file and the target to run machine
    """
    target_feature: str
    file_in: str
    folder_path: FolderCache
    test_size: float = 0.8
    random_state: int = 42


@dataclass
class DatasetProperties:
    """All properties for get the file and pass to the machine
    """
    x: ndarray
    y: ndarray
    x_train: ndarray
    x_test: ndarray
    y_train: ndarray
    y_test: ndarray


@dataclass
class RandomForestJson(MachineJson):
    """Basic parameters for Random Forest machine definition
    """

    def __init__(
            self,
            name_machine: MachineNames,
            type_machine: MachinesTypes,
            machine_json: dict
    ) -> None:
        super().__init__(name_machine=name_machine, type_machine=type_machine)
        self.n_estimators = 1000
        self.random_state = 42
        self.n_jobs = -1
        if "parameters" in machine_json:
            parameters = machine_json["parameters"]
            if "n_estimators" in parameters:
                self.n_estimators = parameters["n_estimators"]
            if "random_state" in parameters:
                self.random_state = parameters["random_state"]
            if "n_jobs" in parameters:
                self.n_jobs = parameters["n_jobs"]

    def __str__(self) -> str:
        return f"machine_name: {
            self.name_machine.value
        } - [ n_estimators: {
            self.n_estimators
        }, random_state: {
            self.random_state
        }, n_jobs: {
            self.n_jobs
        } ]"


@dataclass
class ExtremeGradientBoostJson(MachineJson):
    """Basic parameters for Random Forest machine definition
    """

    def __init__(
            self,
            name_machine: MachineNames,
            type_machine: MachinesTypes,
            machine_json: dict,
    ) -> None:
        super().__init__(name_machine=name_machine, type_machine=type_machine)
        self.n_estimators = -1
        self.max_depth = 0
        self.max_leaves = 0
        if "parameters" in machine_json:
            parameters = machine_json["parameters"]
            if "n_estimators" in parameters:
                self.n_estimators = parameters["n_estimators"]
            if "max_leaves" in parameters:
                self.max_leaves = parameters["max_leaves"]
            if "max_depth" in parameters:
                self.max_depth = parameters["max_depth"]

    def __str__(self) -> str:
        return f"machine_name: {
            self.name_machine.value
        } - [ n_estimators: {
            self.n_estimators
        }, max_depth: {
            self.max_depth
        }, max_leaves: {
            self.max_leaves
        } ]"


@dataclass
class BayesianJson(MachineJson):
    """Basic parameters for Random Forest machine definition
    """

    def __init__(self, name_machine: MachineNames, type_machine: MachinesTypes) -> None:
        super().__init__(name_machine=name_machine, type_machine=type_machine)

    def __str__(self) -> str:
        return f"machine_name: {
            self.name_machine.value
        } - [ ]"


@dataclass
class LassoJson(MachineJson):
    """Basic parameters for Random Forest machine definition
    """

    def __init__(
        self,
        name_machine: MachineNames,
        type_machine: MachinesTypes,
        machine_json: dict,
    ) -> None:
        super().__init__(name_machine=name_machine, type_machine=type_machine)
        self.alpha = 0.1
        if "parameters" in machine_json:
            parameters = machine_json["parameters"]
            if "alpha" in parameters:
                self.alpha = parameters["alpha"]

    def __str__(self) -> str:
        return f"machine_name: {
            self.name_machine.value
        } - [ ]"


@dataclass
class SupportVectorMachineJson(MachineJson):
    """Basic parameters for Random Forest machine definition
    """

    def __init__(
            self,
            name_machine: MachineNames,
            type_machine: MachinesTypes,
            machine_json: dict,
    ) -> None:
        super().__init__(name_machine=name_machine, type_machine=type_machine)
        self.kernel = SupportVectorKernelEnum.LINEAR
        self.c = 1.0
        self.epsilon = 0.1
        if "parameters" in machine_json:
            parameters = machine_json["parameters"]
            if "kernel" in parameters:
                self.kernel = cast_kernel_svm(
                    input_kernel=parameters["kernel"])
            if "c" in parameters:
                self.c = parameters["c"]
            if "epsilon" in parameters:
                self.c = parameters["epsilon"]

    def __str__(self) -> str:
        return f"machine_name: {
            self.name_machine.value
        } - [kernel: {
            self.kernel.value
        }, c: {
            self.c
        }, epsilon: {
            self.epsilon
        } ]"


def build_machine_definition(machine_json: dict) -> MachineJson:
    """Create a machine definition to work with it

   Returns:
        MachineJson: machine definition to operate with it
    """
    if "name" not in machine_json:
        raise NotImplementedError("Not have a name of machine")
    if machine_json["name"] == MachineNames.RFR.value:
        return RandomForestJson(
            name_machine=MachineNames.RFR,
            type_machine=MachinesTypes.R,
            machine_json=machine_json,
        )
    if machine_json["name"] == MachineNames.XGBR.value:
        return ExtremeGradientBoostJson(
            name_machine=MachineNames.XGBR,
            type_machine=MachinesTypes.R,
            machine_json=machine_json,
        )
    if machine_json["name"] == MachineNames.BAR.value:
        return BayesianJson(
            name_machine=MachineNames.BAR,
            type_machine=MachinesTypes.R,
        )
    if machine_json["name"] == MachineNames.LAR.value:
        return LassoJson(
            name_machine=MachineNames.LAR,
            type_machine=MachinesTypes.R,
            machine_json=machine_json,
        )
    if machine_json["name"] == MachineNames.SVR.value:
        return SupportVectorMachineJson(
            name_machine=MachineNames.SVR,
            type_machine=MachinesTypes.R,
            machine_json=machine_json,
        )

    raise NotImplementedError("Not have a name of valid machine")

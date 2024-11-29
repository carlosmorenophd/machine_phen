"""Runner to get variable"""

from src.machines.machine_enums import MachineJson, FileAccessRunnerProperties
from src.machines.machine_data_frame_handler import DataFrameHandler
from src.machines.regression_run import machine_build_regression
from src.metrics.metric_enums import MetricEnum
from src.selection_variables.selection_data_frame_handler import SelectionBestMetric
from src.selection_variables.force_brute import combination_columns_from_data_frame


def selection_force_brute_run(
    file_access_runner: FileAccessRunnerProperties,
    machine_definition: MachineJson,
    metric: MetricEnum = MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR,
) -> None:
    """Run forward selection with force brute
    Args:
            initial_column (_type_): _description_
            metric (MetricEnum, optional): metric to validate machine.
                Defaults to MetricEnum.MEAN_ABSOLUTE_PERCENTAGE_ERROR.
    """
    data_frame = DataFrameHandler(file_access_runner=file_access_runner)
    selection = SelectionBestMetric(metric_to_evaluate=metric)
    df = data_frame.get_data_frame_without_target()
    machine = machine_build_regression(machine_definition=machine_definition)
    for combination in combination_columns_from_data_frame(df):
        data_frame.change_column_from_data_frame(columns_to_keep=combination)
        dataset = data_frame.dataset
        machine.build_machine()
        machine.training(dataset.x_train, dataset.y_train)
        error_metric = machine.test(
            x_test=dataset.x_test,
            y_test=dataset.y_test,
        )
        selection.add_metric_value(
            machine_definition=machine_definition,
            columns=combination,
            metrics=error_metric.metrics,
        )
    df_metric = selection.metric_values
    data_frame.storage_file.save_data_frame_to_csv(
        data_frame=df_metric, prefix="metric_forward_selection",
    )

def selection_evolution_strategy_run() -> None:


from deap import base, creator, tools
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score   


# Configuración del problema
# ... (cargar datos, definir parámetros, etc.)

# Crear la clase de individuo (cromosoma)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Función para generar un individuo aleatorio
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool,   
 n=num_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)   


# Función de evaluación
def evaluate(individual):
    # Crear un nuevo conjunto de entrenamiento con las características seleccionadas
    X_train_selected = X_train[:, np.where(np.array(individual) == 1)[0]]
    # Entrenar un modelo (e.g., Random Forest)
    model = RandomForestClassifier()
    model.fit(X_train_selected, y_train)
    # Evaluar el modelo
    X_test_selected = X_test[:, np.where(np.array(individual) == 1)[0]]
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,

# Registrar los operadores genéticos
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select",   
 tools.selTournament, tournsize=3)   


# Crear la población inicial
pop = toolbox.population(n=30)

# Bucle principal del algoritmo evolutivo
for gen in range(100):
    # ... (evaluación, selección, cruzamiento, mutación)
    # ... (actualizar la población)

# Obtener el mejor individuo
best_ind = tools.selBest(pop, 1)[0]
print(best_ind, best_ind.fitness.values)
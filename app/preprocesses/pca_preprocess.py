from sklearn.decomposition import PCA
from numpy import abs, ndarray, cumsum, sum, arange, where, percentile
from typing import Tuple
import matplotlib.pyplot as plt


class PCA_Preprocess:
    def __init__(
        self,
        data: ndarray,
        target: ndarray,
        feature_names: ndarray,
        is_debug: bool = False,
    ) -> None:
        self.data = data
        self.target = target
        self.is_debug = is_debug
        self.feature_names = feature_names

    def get_transform(
        self,
        n_components: int = None,
        is_search_best_component: bool = False,
        threshold: float = 0.95,
    ) -> ndarray:
        if(is_search_best_component):
            evaluate_pca = PCA()
            evaluate_pca.fit_transform(self.data)
            cumulative_variance = 0
            for i, ratio in enumerate(evaluate_pca.explained_variance_ratio_):
                cumulative_variance += ratio
                if cumulative_variance >= threshold:
                    break
            number_of_pcs = i + 1
            if self.is_debug:
                print(f"Number of component ${number_of_pcs}")
            self.number_of_pcs = number_of_pcs
            pca = PCA(n_components=number_of_pcs)
            return pca.fit_transform(self.data)
        else:
            pca = PCA(n_components=n_components)
            return pca.fit_transform(self.data)

    def evaluate_pca(self, n_components=None, threshold :  float = 0.95) -> Tuple[ndarray, ndarray]:
        if n_components != None and n_components > 0:
            pca = PCA(n_components=n_components)
        else:
            pca = PCA()
        self.pca_x = pca.fit_transform(self.data)
        cumulative_variance = 0
        for i, ratio in enumerate(pca.explained_variance_ratio_):
            cumulative_variance += ratio
            if cumulative_variance >= threshold:
                break
        self.number_of_pcs = i + 1
        weights = pca.components_[0]
        self.most_important_columns = abs(weights).argsort()[::-1]
        self.list_important_features = []
        for index in self.most_important_columns:
            self.list_important_features.append(self.feature_names[index])
        self.eigenvalues = pca.explained_variance_
        if self.is_debug:
            print("Eigenvalues => ", self.eigenvalues)
        self.explained_variance_ratio = pca.explained_variance_ratio_
        if self.is_debug:
            print("Proportion => ", self.explained_variance_ratio)
        # Calcular la proporción acumulada de la varianza explicada
        self.cumulative_explained_variance = cumsum(pca.explained_variance_ratio_)
        if self.is_debug:
            print("Accumulated => ", self.cumulative_explained_variance)
        # Obtener las cargas de cada variable en cada componente
        loadings = pca.components_
        # Normalizar las cargas para obtener la influencia relativa de cada variable en cada componente
        self.normalized_loadings = loadings / sum(abs(loadings), axis=1, keepdims=True)

        self.loadings_by_variable = pca.components_.T

        self.pc_scores = self.pca_x

    def write_to_csv(self, csv_name) -> None:
        import csv

        with open(csv_name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(self.feature_names)
            writer.writerow(self.eigenvalues)
            writer.writerow(self.explained_variance_ratio)
            writer.writerow(self.cumulative_explained_variance)
            print(self.loadings_by_variable.shape)
            for loading in self.loadings_by_variable:
                writer.writerow(loading)

    def outlier_excel(self, xls_name, threshold=1.5, color="0000FF"):
        from openpyxl import Workbook
        from openpyxl.styles import PatternFill

        # Calcular el rango intercuartílico para identificar valores atípicos
        Q1 = percentile(self.pca_x, 25)
        Q3 = percentile(self.pca_x, 75)
        IQR = Q3 - Q1

        # Definir límites para identificar valores atípicos
        lower_limit = Q1 - threshold * IQR
        upper_limit = Q3 + threshold * IQR

        # Encontrar valores atípicos
        indices_filas, indices_columnas = where(
            (self.pca_x < lower_limit) | (self.pca_x > upper_limit)
        )

        # Crear un libro de Excel y agregar una hoja de trabajo
        wb = Workbook()
        ws = wb.active

        # Guardar la matriz original en la hoja de trabajo
        for r_idx, fila in enumerate(self.pca_x, start=1):
            for c_idx, valor in enumerate(fila, start=1):
                ws.cell(row=r_idx, column=c_idx, value=valor)

        # Resaltar valores atípicos en azul
        for r, c in zip(indices_filas, indices_columnas):
            # Sumar 1 para ajustar el índice base 0
            cell = ws.cell(row=r + 1, column=c + 1)
            cell.fill = PatternFill(
                start_color=color, end_color=color, fill_type="solid"
            )

        # Guardar el libro de Excel
        wb.save(xls_name)

    def graph_sedimentation(self):
        # Gráfica de sedimentación (scree plot)
        plt.figure(figsize=(10, 6))
        plt.bar(
            range(1, len(self.eigenvalues) + 1),
            self.eigenvalues,
            alpha=0.8,
            align="center",
            label="Valor propio",
        )
        plt.plot(
            range(1, len(self.cumulative_explained_variance) + 1),
            self.cumulative_explained_variance,
            marker="o",
            linestyle="--",
            color="r",
            label="Varianza acumulada",
        )
        plt.xlabel("Componente Principal")
        plt.ylabel("Valor Propio / Varianza Acumulada")
        plt.title("Análisis de Valores Propios y Gráfica de Sedimentación (Scree Plot)")
        plt.legend()
        plt.show()

    def graph_scores(self):
        # Crear un gráfico de barras para la proporción de varianza explicada
        plt.figure(figsize=(10, 6))
        plt.bar(
            range(1, len(self.explained_variance_ratio) + 1),
            self.explained_variance_ratio,
            alpha=0.8,
            align="center",
        )
        plt.xlabel("Componente Principal")
        plt.ylabel("Proporción de Varianza Explicada")
        plt.title("Proporción de Varianza Explicada por Componente Principal")
        plt.show()

    def graph_influence(self):
        # Crear gráfico de acumulación de varianza explicada
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(
            range(1, len(self.cumulative_explained_variance) + 1),
            self.cumulative_explained_variance,
            marker="o",
        )
        plt.xlabel("Número de Componentes Principales")
        plt.ylabel("Varianza Acumulada Explicada")
        plt.title("Gráfico de Acumulación de Varianza Explicada")
        # Crear gráfico de barras para la influencia de cada variable en el primer componente principal
        plt.subplot(1, 2, 2)
        variables = range(len(self.feature_names))
        plt.bar(variables, self.normalized_loadings[0, :], alpha=0.8)
        plt.xlabel("Variable")
        plt.ylabel("Influencia Relativa")
        plt.title("Influencia de Variables en el Primer Componente Principal")
        plt.tight_layout()
        plt.show()

    def graph_projection(self):
        # Crear un gráfico de doble proyección
        plt.figure(figsize=(10, 5))
        # Gráfico de dispersión para la proyección en los dos primeros componentes principales
        plt.subplot(1, 2, 1)
        plt.scatter(
            self.pca_x[:, 0], self.pca_x[:, 1], c=self.target, cmap="viridis", alpha=0.8
        )
        plt.xlabel("Componente Principal 1 (PC1)")
        plt.ylabel("Componente Principal 2 (PC2)")
        plt.title("Proyección Bidimensional con PCA")
        # Gráfico de barras para mostrar la contribución de cada variable en los dos primeros componentes principales
        plt.subplot(1, 2, 2)
        bar_width = 0.4
        bar_positions = arange(len(self.feature_names))
        plt.bar(
            bar_positions - bar_width / 2,
            self.loadings_by_variable[:, 0],
            width=bar_width,
            label="PC1",
            alpha=0.8,
        )
        plt.bar(
            bar_positions + bar_width / 2,
            self.loadings_by_variable[:, 1],
            width=bar_width,
            label="PC2",
            alpha=0.8,
        )
        plt.xlabel("Variable")
        plt.ylabel("Carga en el Componente Principal")
        plt.title(
            "Contribución de Variables en los Dos Primeros Componentes Principales"
        )
        plt.xticks(bar_positions, self.feature_names, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def graph_outlier(self, threshold=2.5):
        outliers = where(abs(self.pc_scores) > threshold)

        # Visualizar las puntuaciones en un gráfico de dispersión para cada par de componentes principales
        num_components = self.pc_scores.shape[1]

        plt.figure(figsize=(15, 10))

        for i in range(num_components):
            for j in range(i + 1, num_components):
                plt.subplot(
                    num_components - 1,
                    num_components - 1,
                    (i * (num_components - 1)) + j,
                )

                plt.scatter(
                    self.pc_scores[:, i],
                    self.pc_scores[:, j],
                    edgecolors="k",
                    c="none",
                    marker="o",
                    alpha=0.8,
                )
                plt.scatter(
                    self.pc_scores[outliers[0], i],
                    self.pc_scores[outliers[0], j],
                    edgecolors="r",
                    facecolors="r",
                    marker="o",
                    s=200,
                    label="Outliers",
                )

                plt.xlabel(f"PC{i + 1}")
                plt.ylabel(f"PC{j + 1}")
                plt.title(f"PC{i + 1} vs PC{j + 1}")

        # Ajustar diseño del layout
        plt.tight_layout()
        plt.savefig("outlier.png")

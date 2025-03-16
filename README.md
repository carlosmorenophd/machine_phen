# A Machine Learning Process Orchestrated by a Genetic Algorithm

This project utilizes a genetic algorithm to optimize a machine learning pipeline for predicting phenotypic data.

**Recommended Environment:**

* **Operating System:** Ubuntu 24.04
* **Hardware:**
    * 128GB RAM
    * 256GB Swap Space
    * 1TB SSD Storage
    * Ryzen 5800X or later CPU

## Running the Project

The project is containerized using Docker Compose. Follow these steps to run it:

1.  **Environment Configuration:**
    * Create a new `.env` file in the project's root directory.
    * You can use `.env.example` as a template.
    * Modify the following variables in `.env`:
        * `REDIS_URL`: Specifies the Redis server URL (e.g., `redis://redis:6379/0`).
        * `PATH_TO_CACHE`: Defines the path where data files and cache will be stored. **Ensure this path is correctly set as it's crucial for the project's operation.**

    ```
    REDIS_URL=redis://redis:6379/0
    PATH_TO_CACHE=path/to/store/files
    ```

2.  **Building Docker Images:**
    * Open a terminal in the project's root directory.
    * Execute the command: `docker compose build`

3.  **Running the Containers:**
    * Execute the command: `docker compose up -d`

## Example Usage

The system provides functionality to run tests and identify the optimal machine learning model for phenotypic data prediction.

**Instructions:**

1.  **Dataset Placement:**
    * Copy your dataset (in CSV format) to the directory specified by the `PATH_TO_CACHE` variable in your `.env` file.

2.  **Attaching to the Container:**
    * Open a terminal.
    * Execute the command: `docker exec -it phen_machine /bin/bash`

3.  **Running the Genetic Algorithm:**
    * Inside the container's bash prompt, execute the following command:

    ```bash
    python tasks_test.py regression_genetic name_of_file.csv name_column_feature_to_predict
    ```

    * Replace:
        * `name_of_file.csv`: with the actual name of your CSV dataset file.
        * `name_column_feature_to_predict`: with the name of the column in your dataset that you want to predict.
```





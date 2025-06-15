# fairness-benchmark

Name Pending.

## References

Portions of Code (/fairness_benchmark/metrics.py) are sourced from [Fair_Fairness_Benchmark](https://github.com/ahxt/fair_fairness_benchmark).

Copies of the above licenses (MIT) can be found in /license/

## Installation

Prerequisites: [Poetry](https://python-poetry.org/). Python: 3.12.7.
[comment]: # TODO: Add in repo download.
Either Clone or Fork this project.

``` Bash
git clone git@github.com:broldfield/fairness-benchmark.git
```

In the project directory >

``` Bash
poetry install
```

## Configuration

Configuration is handled by args in the command line.

| Arg            | Description                                                                 | Variations                                |
| -------------- | --------------------------------------------------------------------------- | ----------------------------------------- |
| --task         | The task the tool will run.                                                 | process, benchmark, config                        |
| --dataset      | What dataset to load.                                                       | adult, bank, compas, german, meps, none    |
| --dataset_path | If the --dataset arg is none, add in a path to a custom dataset.            | fairness_benchmark/data/dataset/my_custom |
| --sensitive    | The Sensitive Attribute                                                     | sex, age, income, race, RACE                    |
| --preprocess   | The Fairness Preprocessing Technique to be used.                            | dir, lfr, op, rw                    |

Additional args to be added:

- n_samples to reduce data size.
- mlp layers

### Task

The Task arg determines whether the tool will either:

- Process the dataset (Read the Dataset, Apply the Fairness Preprocessing techniques, Save the dataset)
- Benchmark a saved dataset using the Model selected from --model.
- Config uses the tasks set in `tasks.yaml` to batch run process or benchmark tasks.

### Setting tasks.yaml

When running the program with `--tasks "config"`, edit the `tasks.yaml` file in the project root to include what tasks to run. \
The yaml file has 2 items:

- run: Add a job into the array to run it. The array is ordered left to right, so include process jobs before benchmark jobs.
- jobs: Each job has a name to be added to the run item. The jobs contain fields that map to the args under \#Configuration

Remember to correctly indent in yaml.

``` yaml
# Add a job to this array. Job: job_1.1 will run first, then job_1.2.
run: ["job_1.1", "job_1.2"]
jobs:
  # Create a Job
  job_1.1:
    task: "process"
    dataset: "german"
    preprocess: "rw"
    sensitive: "sex"
  job_1.2:
    task: "benchmark"
    dataset: "german"
    preprocess: "rw"
    sensitive: "sex"
    model: "lr"
```

## Example

For config (Easiest option):

``` Bash
poetry run python src/fairness_benchmark/main.py --task "config"
```

Otherwise, for process or benchmark:

``` Bash
poetry run python src/fairness_benchmark/main.py --task "process" --preprocess "rw" --sensitive "sex" --dataset "german"
```

After a running a `--task "process`, the Original and Processed datasets are generated in `/data/processed_dataset/`. These are then loaded into `--task "benchmark`.
Keep the args the same between the different tasks, just add in `--model "model_name"` and the same dataset will be used.

``` Bash
poetry run python src/fairness_benchmark/main.py --task "benchmark" --preprocess "rw" --sensitive "sex" --dataset "german" --model "nb"
```

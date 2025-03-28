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
| --task         | The task the tool will run.                                                 | process, benchmark                        |
| --dataset      | What dataset to load.                                                       | adult, bank, compas, german, meps, none    |
| --dataset_path | If the --dataset arg is none, add in a path to a custom dataset.            | fairness_benchmark/data/dataset/my_custom |
| --target       | The Target Attribute for prediction                                         | income                                    |
| --sensitive    | The Sensitive Attribute                                                     | sex, age, income, race, RACE                    |
| --preprocess   | The Fairness Preprocessing Technique to be used.                            | dir, lfr, op, rw                    |

Additional args to be added:

- n_samples to reduce data size.
- mlp layers

### Task

The Task arg determines whether the tool will either:

- Process the dataset (Read the Dataset, Apply the Fairness Preprocessing techniques, Save the dataset)
- Benchmark a saved dataset using the Model selected from --model.

## Example

``` Bash
poetry run python src/fairness_benchmark/main.py --task "process" --preprocess "rw" --sensitive "sex" --dataset "german" --target "credit"
```

After a running a `--task "process`, the Original and Processed datasets are generated in `/data/processed_dataset/`. These are then loaded into `--task "benchmark`.
Keep the args the same between the different tasks, just add in `--model "model_name"` and the same dataset will be used.

``` Bash
poetry run python src/fairness_benchmark/main.py --task "benchmark" --preprocess "rw" --sensitive "sex" --dataset "german" --target "credit" --model "nb"
```

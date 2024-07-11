import argparse
import os
from datetime import datetime
import optuna
import torch
import yaml

from my_utils import open_tensorboard, copy_file


parser = argparse.ArgumentParser(description="Deep learning with primacy bias")
run_id = datetime.now().strftime('%b_%d_%H_%M_%S')
parser.add_argument("--tune", action="store_true",
                    help="Runs the main script using optuna to tune hyperparameters. Note that this means a different "
                         "config file that specifies tuning configurations has to be used")
parser.add_argument("--config", type=str, default="data/config.yaml",
                    help="Path of the file that contains the hyperparameters. Refer to data/config.yaml and "
                         "data/hp_tuning.yaml for an example config file.")
parser.add_argument("--run-id", type=str, default=run_id,
                    help="Name of the current run (results are stored at data/results/<run-id>). The default run-id is "
                         "based on time and date (Month_date_Hours_minutes_seconds).")
args = parser.parse_args()


def config_from_trial(hyperparameters, trial):
    """
    Samples hyperparameters for hyperparameters to be tuned, and merges the samples with the set hyperparameters.
    Parameters
    ----------
    trial : optuna.Trial
    hyperparameters : dict
        A hyperparameter config.
    """
    run_config = hyperparameters.get('set', {})
    for hp_name, hp_info in hyperparameters['tunable'].items():
        suggest_func = getattr(trial, f"suggest_{hp_info['type']}")
        if suggest_func is not None and callable(suggest_func):
            run_config[hp_name] = suggest_func(hp_name, **hp_info['arguments'])
    return run_config


def main(run_config, trial=None, save_dir=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Runs an experiment named `run_tag` with hyperparameters from `run_config`. Results are stored in `log_dir`. Returns
    a tuple of training losses and testing losses over time. This function returns a tuning objective when tuning
    (e.g. an accuracy to maximize) or a tuple with the complete results otherwise.

    Parameters
    ----------
    run_config : dict
        Dictionary that contains hyperparameter values or tuning information when tuning.
    trial : optuna.Trial
    save_dir : str
    device : str

    Returns
    -------
    typing.Union[float, tuple]
    """

    is_tuning = trial is not None
    run_config = config_from_trial(run_config, trial) if is_tuning else run_config
    tag = f"Tuning trial {trial.number}" if is_tuning else "Post tuning"

    if is_tuning:  # Return tuning objective
        return 0
    else:  # Return results
        return [], []


if __name__ == "__main__":

    base_output_path = os.path.join("data", "results", f"{args.run_id}")
    tensorboard_dir = os.path.join(base_output_path, "tensorboard")
    print(f"Saving results to: {base_output_path}")
    open_tensorboard(tensorboard_dir)

    # Find the best hyperparameters through optuna
    if args.tune:
        copy_file(args.config, os.path.join(base_output_path))
        with open(args.config, 'r') as file:
            tuning_config = yaml.safe_load(file)
        study = optuna.create_study(
            direction="maximize",
            storage="sqlite:///data/results/" + args.run_id + "/tuning.db",
            load_if_exists=True,
            # sampler=optuna.samplers.GridSampler,
            sampler=optuna.samplers.RandomSampler()
        )
        
        study.optimize(lambda trial: main(tuning_config, trial=trial), n_trials=tuning_config['n_trials'])
        best_trial = study.best_trial
        config = best_trial.params
        if tuning_config.get('set') is not None:
            config.update(tuning_config['set'])

    # Or use the hyperparameters specified in the config file.
    else:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)

    # Save the config in the results directory to ensure we can find the used config later on.
    with open(os.path.join(base_output_path, "hyperparameters.yaml"), 'w') as file:
        yaml.dump(config, file)

    # Run main with the best, or specified parameters.
    main(
        config,
        save_dir=os.path.join(base_output_path, "models"),
    )

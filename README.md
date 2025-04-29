![alt text](assets/banner.png)

# R.L.M's Machine Learning Recipe
Welcome to this incredible repo! With this module, you can utilize my code, which may not be the best, to write your 
own code, which may also not be the best, all in the pursuit of training an AI to generate code that most certainly is
not the best. (Disclaimer: That was a joke. The code isn't that amazing. But hey, you could always try training an AI to 
master digit recognition on the MNIST dataset! Even better, Right?) 

## Installation
First install pytorch as discussed in the [installation guide](https://pytorch.org/get-started/locally/). 
Then, install the requirements using pip:
```bash
pip install .
```

## Usage

### Running experiments
The code should be run using one of the configurations in the [configs](configs) folder. You can run the code using
the following command:
```bash
python -m scripts.main --config-name=[name of your config]
```
You can use Hydra's [override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/) to override the 
configurations from the command line. For example, to change the optimizer to AdamW, you can use:
```bash
python -m scripts.main --config-name=[name of your config] optimizer=adamw 
```

### Running multiple experiments at once
Alternatively, you can run multiple experiments with one command as follows:
```bash
python -m scripts.main [key]=[value1],[value2],...,[valueN] --multirun
```

### Running hyperparameter tuning
To run hyperparameter tuning, you can override the tuning configuration as follows:
```bash
python -m scripts.main tuning=sample_search
```

By specifying the `hydra.sweeper.storage` option in the hp_tuning configuration, you can choose the storage backend for
optuna. During the tuning process, this will save the results of the trials to the specified database. You can 
load a previously saved database by specifying the `hydra.sweeper.storage` and 'hydra.sweeper.study_name' options in the
hp_tuning configuration. This will continue the tuning process from the specified study. This database can
be accessed using the optuna dashboard, which can be started using the following command:
```bash
optuna-dashboard sqlite:///path/to/database.db
```

To run multiple programs at once for the same hyperparameter search, simply run the script with the same config 
multiple times.

#### Distributed hyperparameter tuning
By providing access to the database by hosting it on a server, it should be possible to run multiple programs at once
on different machines. However, I have not yet figured out how to do this exactly.

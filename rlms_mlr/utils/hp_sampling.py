from optuna import Trial

def sample_hp(trial: Trial, config: dict):
    sampled_hp = {}
    for key, value in config.items():
        suggest_fn = getattr(trial, f'suggest_{value["type"]}')
        sampled_hp[key] = suggest_fn(key, **value["args"])

    return sampled_hp
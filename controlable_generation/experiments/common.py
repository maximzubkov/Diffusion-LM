import json
from os.path import join

from improved_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults


def model_and_diffusion(init_emb: str):
    config_path = join(init_emb, "training_args.json")
    with open(config_path, 'rb', ) as f:
        training_args2 = json.load(f)
    training_args2['sigma_small'] = True
    training_args2['diffusion_steps'] = 200
    temp_dict = model_and_diffusion_defaults()
    temp_dict.update(training_args2)
    return create_model_and_diffusion(
        **temp_dict
    )

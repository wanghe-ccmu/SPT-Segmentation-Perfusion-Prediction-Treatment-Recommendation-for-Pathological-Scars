"""Model loading utilities for keloid blood perfusion prediction."""

import importlib
from models.base_model import BaseModel


def find_model_using_name(model_name):
    """Import the module ``models/[model_name]_model.py``."""
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        raise ImportError(
            "Model class not found: expected %s in %s.py"
            % (target_model_name, model_filename)
        )

    return model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Create the configured blood perfusion model."""
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("[Blood Perfusion] Model: %s" % type(instance).__name__)
    return instance

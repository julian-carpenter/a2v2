"""Public API for A2V2's Animal2Vec 1.0 reproduction baseline.

The five implementation modules remain available for researchers who need
lower-level control. These exports cover the common path: load a recipe, build
one of the two task models, or run inference from a native checkpoint.
"""

# Mathematics: these exports define the small public set
# {configuration, pretraining model, fine-tuning model, inference runner};
# helper functions remain in their five responsibility-based modules.
# Interpretation: normal users see one stable entry surface, while researchers
# can read or import lower-level equations from the file that owns them.
from .config import Animal2VecConfig, load_config
from .model import Animal2VecFineTuningModel, Animal2VecPretrainingModel
from .workflows import InferenceResult, InferenceRunner

__all__ = [
    "Animal2VecConfig",
    "Animal2VecFineTuningModel",
    "Animal2VecPretrainingModel",
    "InferenceResult",
    "InferenceRunner",
    "load_config",
]

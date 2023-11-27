from .onelvae import (
    OneLatentVAEVanilla,
    OneLatentVAEUniform,
    OneLatentVAEFixedVariance,
)

vae_models = {
    "1lvae-vanilla": OneLatentVAEVanilla,
    "1lvae-fixedvar" : OneLatentVAEFixedVariance,
    "1lvae-uniform": OneLatentVAEUniform,
}

vae_config = {
    "1lvae-vanilla": {"M": 128, "N": 64},
    "1lvae-fixedvar": {"M": 128, "N": 64},
    "1lvae-uniform": {"M": 128, "N": 64},
}


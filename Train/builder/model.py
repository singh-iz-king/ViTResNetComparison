import torch
from Model.vit import VitClassifierWrapper, MAEWrapper, SimCLRWrapper


def build_model(cfg):
    m = cfg["model"]
    kind = m["kind"]

    if kind == "vit_classifier":
        model = VitClassifierWrapper(m["embedding_dim"], m["patch_size"], 200)

        init_path = m.get("init_encoder_from", None)
        if init_path:
            enc_state = torch.load(init_path, map_location="cpu")
            model.Encoder.load_state_dict(enc_state, strict=True)

        return model

    if kind == "mae":
        model = MAEWrapper(m["embedding_dim"], m["patch_size"])
        return model

    if kind == "simclr_vit":
        model = SimCLRWrapper(m["embedding_dim"], m["patch_size"])
        return model

    raise ValueError(f"Could not find model of kind: {kind}")

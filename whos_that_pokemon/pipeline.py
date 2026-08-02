import torch

from . import config
from .models import UNet, PokemonClassifier


def load_pipeline(num_classes, device, manifest):
    seg_model = UNet(base=32).to(device)
    seg_model.load_state_dict(torch.load(config.SEG_CHECKPOINT, map_location=device))
    seg_model.eval()

    classifier = PokemonClassifier(num_classes=num_classes).to(device)
    classifier.load_state_dict(torch.load(config.CLS_CHECKPOINT, map_location=device))
    classifier.eval()

    pokemon_ids_unsorted = list(set(m["pokemon_id"] for m in manifest))
    id_to_label = {pid: i for i, pid in enumerate(pokemon_ids_unsorted)}
    label_to_name = {
        lbl: next(m["name"] for m in manifest if m["pokemon_id"] == pid)
        for pid, lbl in id_to_label.items()
    }

    return seg_model, classifier, label_to_name


def make_predict_fn(seg_model, classifier, label_to_name, device):
    def predict_pokemon(img_tensor):
        with torch.no_grad():
            seg_logits = seg_model(img_tensor.unsqueeze(0).to(device))
            pred_mask = (torch.sigmoid(seg_logits) > 0.5).float()

            cls_logits = classifier(pred_mask)
            pred_label = cls_logits.argmax(1).item()

        return label_to_name[pred_label], pred_mask.cpu().squeeze().numpy()

    return predict_pokemon
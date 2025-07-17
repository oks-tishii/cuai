# --- Configuration ---
from pathlib import Path

from dataset import MVTecDataset
from models import PatchCore


MODEL_NAME = "patchcore"
DATASET_NAME = "screw"
BACKBONE = "wide_resnet50_2"
F_CORESET = 1.0

# --- Paths ---
EXPORT_DIR = Path("./exports")


def train_and_save():
    # 1. Initialize Model
    print(f"Initializing {MODEL_NAME} model with backbone {BACKBONE}...")
    model = PatchCore(
        f_coreset=F_CORESET,
        backbone_name=BACKBONE,
    )

    # 2. Load Data
    print(f"Loading '{DATASET_NAME}' dataset...")
    train_ds, _ = MVTecDataset(DATASET_NAME).get_dataloaders()

    # 3. Train Model
    print("Training model...")
    model.fit(train_ds)
    print("Training complete.")

    # 4. Export Model (to default location)
    save_name = f"{MODEL_NAME}_{DATASET_NAME}"
    print(f"Exporting model to '{EXPORT_DIR}/{save_name}.pt'...")
    # The export method saves a scripted model, which is what we want.
    model.export(save_name)
    print("Export complete.")


if __name__ == "__main__":
    train_and_save()

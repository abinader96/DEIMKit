import requests
import zipfile
import os
from deimkit import Trainer, Config, configure_dataset, load_model, list_models
from loguru import logger


# Simple function to download and extract a zip file
def download_and_unzip(url, extract_to="./"):
    """Download a zip file and extract its contents"""
    # Download the file
    print(f"Downloading {url}...")
    response = requests.get(url)
    zip_path = os.path.join(extract_to, os.path.basename(url))

    # Save the zip file
    os.makedirs(extract_to, exist_ok=True)
    with open(zip_path, "wb") as f:
        f.write(response.content)

    # Extract the zip file
    print(f"Extracting to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    print("Download and extraction complete!")
    return extract_to


logger.info("Downloading and extracting COCO8 dataset...")
download_and_unzip(
    "https://github.com/dnth/DEIMKit/releases/download/v0.1.0/coco8.zip", "./data"
)

logger.info("Listing models supported by DEIMKit...")
logger.info(list_models())

logger.info("Configuring model and dataset...")
conf = Config.from_model_name("deim_hgnetv2_n")

conf = configure_dataset(
    config=conf,
    image_size=(320, 320),
    train_ann_file="./data/coco8-converted/train_coco.json",
    train_img_folder="./data/coco8-converted",
    val_ann_file="./data/coco8-converted/val_coco.json",
    val_img_folder="./data/coco8-converted",
    train_batch_size=2,
    val_batch_size=2,
    num_classes=81,
    output_dir="./outputs/coco8/deim_hgnetv2_n",
)

logger.info("Training model...")
trainer = Trainer(conf)

trainer.fit(epochs=3, save_best_only=True)

logger.info("Loading pretrained model...")
model = load_model("deim_hgnetv2_n", image_size=(640, 640))

logger.info("Testing predictions...")
predictions = model.predict("./data/coco8-converted/000000000009.jpg")

logger.info("Loading best model from training...")
model = load_model(
    "deim_hgnetv2_n",
    image_size=(320, 320),
    checkpoint="./outputs/coco8/deim_hgnetv2_n/best.pth",
)

logger.info("Testing predictions...")
predictions = model.predict("./data/coco8-converted/000000000009.jpg")

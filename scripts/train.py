from deimkit import Trainer, Config, configure_dataset

conf = Config.from_model_name("deim_hgnetv2_s")

conf = configure_dataset(
    config=conf,
    image_size=(640, 640),
    train_ann_file="/home/dnth/Desktop/DEIMKit/dataset_collections/CSGO videogame.v2-release.coco/train/_annotations.coco.json",
    train_img_folder="/home/dnth/Desktop/DEIMKit/dataset_collections/CSGO videogame.v2-release.coco/train",
    val_ann_file="/home/dnth/Desktop/DEIMKit/dataset_collections/CSGO videogame.v2-release.coco/valid/_annotations.coco.json",
    val_img_folder="/home/dnth/Desktop/DEIMKit/dataset_collections/CSGO videogame.v2-release.coco/valid",
    train_batch_size=16,
    val_batch_size=16,
    num_classes=3,
    output_dir="./outputs/csgo-videogame/deim_hgnetv2_s_30ep_640x640",
)

trainer = Trainer(conf)

trainer.fit(epochs=30, save_best_only=True)

from deimkit import Trainer, Config, configure_dataset

conf = Config.from_model_name("deim_hgnetv2_n")

conf = configure_dataset(
    config=conf,
    image_size=(640, 640),
    train_ann_file="/home/dnth/Desktop/DEIMKit/dataset_collections/mouse stem cells.v1i.coco/train/_annotations.coco.json",
    train_img_folder="/home/dnth/Desktop/DEIMKit/dataset_collections/mouse stem cells.v1i.coco/train",
    val_ann_file="/home/dnth/Desktop/DEIMKit/dataset_collections/mouse stem cells.v1i.coco/valid/_annotations.coco.json",
    val_img_folder="/home/dnth/Desktop/DEIMKit/dataset_collections/mouse stem cells.v1i.coco/valid",
    train_batch_size=4,
    val_batch_size=4,
    num_classes=2,
    output_dir="./outputs/mouse-stem-cells/deim_hgnetv2_n_3000ep",
)


trainer = Trainer(conf)

trainer.fit(epochs=3000, save_best_only=True)

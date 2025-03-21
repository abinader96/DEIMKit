from deimkit import Trainer, Config, configure_dataset

conf = Config.from_model_name("deim_hgnetv2_s")

conf = configure_dataset(
    config=conf,
    image_size=(352, 640),
    train_ann_file="/home/dnth/Desktop/DEIMKit/dataset_collections/Construction Site Safety.v30-raw-images_latestversion.coco/train/_annotations.coco.json",
    train_img_folder="/home/dnth/Desktop/DEIMKit/dataset_collections/Construction Site Safety.v30-raw-images_latestversion.coco/train",
    val_ann_file="/home/dnth/Desktop/DEIMKit/dataset_collections/Construction Site Safety.v30-raw-images_latestversion.coco/valid/_annotations.coco.json",
    val_img_folder="/home/dnth/Desktop/DEIMKit/dataset_collections/Construction Site Safety.v30-raw-images_latestversion.coco/valid",
    train_batch_size=16,
    val_batch_size=16,
    num_classes=26,
    output_dir="./outputs/construction-site-safety/deim_hgnetv2_s_50ep_352x640",
)


trainer = Trainer(conf)

trainer.fit(epochs=50, save_best_only=True)

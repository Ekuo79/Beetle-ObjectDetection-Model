from rfdetr import RFDETRBase
model = RFDETRBase()

dataset_path = '/blue/hulcr/share/eric.kuo/Beetle_classifier/Data/00_Preprocessed_composite_images/rfdetr_dataset'
output_path = '/blue/hulcr/eric.kuo/rfdetr/run3'

model.train(
    dataset_dir=dataset_path,
    epochs=30,
    batch_size=6,
    grad_accum_steps=1,
    lr=1e-4,
    output_dir=output_path,
)
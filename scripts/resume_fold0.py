from spheriq.train import train_on

train_on(
    datasets=[{'name': 'odi', 'stereo': False}],
    epochs=40,
    continue_from=20,
    patch_size=32,
    grid_size=16,
    batch_size=2,
    cpu_workers=12,
    pretrained=True,
    val_tta_angles=[0],
    artifact_aug_prob=0.0,
    num_folds=5,
    fold_index=1,
)

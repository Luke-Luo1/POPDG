import argparse

def parse_train_opt():
    parser = argparse.ArgumentParser(description="Train the model with specified parameters.")

    # Environment settings
    env_group = parser.add_argument_group('Environment and Path Settings')
    env_group.add_argument("--wandb_pj_name", type=str, default="POPDG", help="project name")
    env_group.add_argument("--project", default="experiments/train", help="Base directory for saving training outputs.")
    env_group.add_argument("--exp_name", default="1", help="Experiment name, used for saving outputs within the project directory.")
    env_group.add_argument("--data_path", type=str, default="data/", help="Path to the raw data.")
    env_group.add_argument("--processed_data_dir", type=str, default="data/dataset_backups/", help="Path to save/load processed dataset backups.")
    env_group.add_argument("--render_dir", type=str, default="renders/", help="Directory to save sample renders.")

    # Training settings
    train_group = parser.add_argument_group('Training Settings')
    train_group.add_argument("--feature_type", type=str, default="jukebox", help="Type of features to use for training the model.")
    train_group.add_argument("--batch_size", type=int, default=128, help="Number of samples per batch.")
    train_group.add_argument("--epochs", type=int, default=2000, help="Total number of training epochs.")

    # Advanced settings
    advanced_group = parser.add_argument_group('Advanced Settings')
    advanced_group.add_argument("--force_reload", action="store_true", help="Force reloading the datasets, ignoring cached versions.")
    advanced_group.add_argument("--no_cache", action="store_true", help="Do not cache the loaded datasets for future runs.")
    advanced_group.add_argument("--save_interval", type=int, default=100, help='Interval (in epochs) at which to save model checkpoints.')
    advanced_group.add_argument("--ema_interval", type=int, default=1, help='ema every x steps')
    advanced_group.add_argument("--checkpoint", type=str, default="", help='trained checkpoint path (optional)')

    opt = parser.parse_args()
    return opt

def parse_test_opt():
    parser = argparse.ArgumentParser(description="Configure and run the model testing.")

    # Environment settings
    env_group = parser.add_argument_group('Environment and Path Settings')
    env_group.add_argument("--processed_data_dir", type=str, default="data/dataset_backups/", help="Path where processed dataset backups are stored.")
    env_group.add_argument("--render_dir", type=str, default="renders/", help="Directory where rendered outputs will be saved.")
    env_group.add_argument("--music_dir", type=str, default="data/test/wavs", help="Directory containing input music files for testing.")
    env_group.add_argument("--motion_save_dir", type=str, default="eval/motions", help="Directory where generated motion files will be saved if --save_motions is used.")

    # Testing settings
    test_group = parser.add_argument_group('Test Execution Settings')
    test_group.add_argument("--feature_type", type=str, default="jukebox", help="Type of features to use for the model testing.")
    test_group.add_argument("--out_length", type=float, default=10.0, help="Maximum length of the output in seconds.")
    test_group.add_argument("--checkpoint", type=str, default="checkpoint.pt", help="Path to the model checkpoint to be used for testing.")

    # Cache settings 
    cache_group = parser.add_argument_group('Feature and Caching Settings')
    cache_group.add_argument("--save_motions", action="store_true", help="Enable saving the generated motions for further evaluation.")
    cache_group.add_argument("--cache_features", action="store_true", help="Enable caching of computed features for reuse.")
    cache_group.add_argument("--use_cached_features", action="store_true", help="Use precomputed features instead of recalculating.")
    cache_group.add_argument("--feature_cache_dir", type=str, default="cached_features/", help="Directory to save/load cached features.")
    cache_group.add_argument("--no_render", action="store_true", help="Disable video rendering after testing.")

    opt = parser.parse_args()
    return opt

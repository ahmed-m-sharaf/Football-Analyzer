import argparse
import os
from dotenv import load_dotenv
load_dotenv()
import yaml
from roboflow import Roboflow
from ultralytics import YOLO


def train_yolo(
    api_key: str | None = None,
    model_name: str = "yolo26x.pt",
    epochs: int = 100,
    batch_size: int = 3,
    patience: int = 20,
    imgsz: int = 1280,
    device: str | None = None,
) -> None:
    """
    Downloads dataset from Roboflow and trains/fine-tunes a YOLO model.

    Parameters
    ----------
    api_key : str | None
        Roboflow API key. If not provided, it will check the ROBOFLOW_API_KEY environment variable.
    model_name : str
        The name of the YOLO model checkpoint or YAML configuration file.
    epochs : int
        Number of epochs to train the model.
    batch_size : int
        Training batch size.
    patience : int
        Patience for early stopping during training.
    imgsz : int
        Size of input images.
    device : str | None
        Target device for training (e.g. '0', 'cpu', 'cuda').
    """
    # Resolve project root and datasets directory relative to this script
    # __file__ is in project_root/src/detection/
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    datasets_dir = os.path.join(project_root, "datasets")
    dataset_location = os.path.join(
        datasets_dir, "football-players-detection-20"
    )

    # 1. Download Dataset
    if os.path.exists(dataset_location):
        print(f"Dataset already exists at {dataset_location}. Skipping download.")
    else:
        # Check API key
        roboflow_api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not roboflow_api_key:
            raise ValueError(
                "Roboflow API key is required. Please provide it via --api-key "
                "or set the ROBOFLOW_API_KEY environment variable."
            )

        print(f"Creating datasets directory at {datasets_dir}...")
        os.makedirs(datasets_dir, exist_ok=True)

        original_cwd = os.getcwd()
        try:
            print("Changing directory to datasets directory for download...")
            os.chdir(datasets_dir)

            print("Connecting to Roboflow...")
            rf = Roboflow(api_key=roboflow_api_key)
            project = rf.workspace("roboflow-jvuqo").project(
                "football-players-detection-3zvbc"
            )
            version = project.version(20)

            print("Downloading dataset version 20 (yolo26)...")
            dataset = version.download("yolo26")
            dataset_location = dataset.location
        finally:
            os.chdir(original_cwd)

    # 2. Update data.yaml
    data_yaml_path = os.path.join(dataset_location, "data.yaml")
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")

    print(f"Updating dataset paths in {data_yaml_path}...")
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Standardize/fix relative paths to train and val images in dataset folder
    data["train"] = "../train/images"
    data["val"] = "../valid/images"

    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data, f)

    # 3. Train Model
    print(f"Initializing YOLO model: {model_name}")
    model = YOLO(model_name)

    print(f"Starting training on dataset {dataset_location}...")
    # Training results will be saved under the 'runs' directory relative to the CWD.
    # Typically, running this from the project root will create runs/ at the project root.
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        patience=patience,
        imgsz=imgsz,
        device=device,
        close_mosaic=15,
        cos_lr=True,
        cache=True,
        plots=True,
    )
    print("Training process finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO model for football player detection."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Roboflow API key. Overrides the ROBOFLOW_API_KEY environment variable.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26x.pt",
        help="YOLO model checkpoint or configuration name (default: yolo26x.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=3,
        help="Batch size (default: 3).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience in epochs (default: 20).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Input image resolution size (default: 1280).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device for execution (e.g. cpu, cuda, or device index 0,1).",
    )

    args = parser.parse_args()
    train_yolo(
        api_key=args.api_key,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        patience=args.patience,
        imgsz=args.imgsz,
        device=args.device,
    )

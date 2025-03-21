# Prediction interface for Cog ⚙️
# https://cog.run/python
import os
from cgi import print_directory

from cog import BasePredictor, Input, Path

from oemer import MODULE_PATH
from oemer.build_system import logger
from oemer.ete import CHECKPOINTS_URL, download_file
from predict_with_bboxes import predict_bboxes


class Predictor(BasePredictor):
    def setup(self) -> None:
        Predictor.print_directory_tree(MODULE_PATH)

        chk_path = os.path.join(MODULE_PATH, "checkpoints/unet_big/model.onnx")
        if not os.path.exists(chk_path):
            logger.warn("No checkpoint found in %s", chk_path)
            os.makedirs(os.path.join(MODULE_PATH, "checkpoints/unet_big"), exist_ok=True)
            os.makedirs(os.path.join(MODULE_PATH, "checkpoints/seg_net"), exist_ok=True)
            for idx, (title, url) in enumerate(CHECKPOINTS_URL.items()):
                logger.info(f"Downloading checkpoints ({idx + 1}/{len(CHECKPOINTS_URL)})")
                save_dir = "unet_big" if title.startswith("1st") else "seg_net"
                save_dir = os.path.join(MODULE_PATH, "checkpoints", save_dir)
                save_path = os.path.join(save_dir, title.split("_")[1])
                download_file(title, url, save_path)

    @staticmethod
    def print_directory_tree(start_path):
        """
        Recursively prints the directory tree starting from startpath.
        Directories are shown with a trailing slash.
        """
        for root, dirs, files in os.walk(start_path):
            # Calculate the depth (number of subdirectories)
            level = root.replace(start_path, "").count(os.sep)
            indent = "." * 4 * level
            # Print the current directory name
            print(f"{indent}{os.path.basename(root)}/")
            # Increase indentation for files and subdirectories
            subindent = "." * 4 * (level + 1)
            for f in files:
                print(f"{subindent}{f}")

    def predict(
        self,
        image: Path = Input(description="Input image"),
        deskew: bool = Input(description="Whether to deskew the image", default=False),
    ) -> Path:
        """Run a single prediction on the model"""
        return predict_bboxes(str(image.absolute()), deskew)

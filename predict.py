# Prediction interface for Cog ⚙️
# https://cog.run/python
import os

from cog import BasePredictor, Input, Path

from oemer.predict_with_bboxes import predict_bboxes


class Predictor(BasePredictor):
    def predict(
        self,
        image: Path = Input(description="Input image"),
        min_pixels: int = Input(description="Minimum pixel size", default=1_250_000),
        max_pixels: int = Input(description="Maximum pixel size", default=1_750_000),
        deskew: bool = Input(description="Whether to deskew the image", default=False),
    ) -> Path:
        """Run a single prediction on the model"""
        os.environ["min_pixels"] = str(min_pixels)
        os.environ["max_pixels"] = str(max_pixels)
        return predict_bboxes(str(image.absolute()), deskew)

# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path

from oemer.predict_with_bboxes import predict_bboxes


class Predictor(BasePredictor):
    def predict(
        self,
        image: Path = Input(description="Input image"),
        deskew: bool = Input(description="Whether to deskew the image", default=False),
    ) -> Path:
        """Run a single prediction on the model"""
        return predict_bboxes(str(image.absolute()), deskew)

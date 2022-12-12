import unittest
import os
import cv2

from pvdn.detection.inference import Runner

THIS_DIR = os.path.dirname(__file__)


class TestDetectionInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        yaml_path = os.path.join(THIS_DIR, "..", "pvdn", "detection", "BestBlobDetectorParameters.yaml")
        weights_path = os.path.join(THIS_DIR, "..", "pvdn", "detection", "weights_pretrained_optimized.pt")
        img_path = os.path.join(THIS_DIR, "data", "047433.png")
        cls.runner = Runner(
            yaml=yaml_path,
            weights=weights_path,
            device="cpu",
            bbox_size=(64, 64)
        )
        cls.test_img = cv2.imread(img_path, 0)
    
    def test_inference(self):
        preds = self.runner.infer(self.test_img)
        self.assertIsNotNone(preds)


if __name__ == "__main__":
    unittest.main()
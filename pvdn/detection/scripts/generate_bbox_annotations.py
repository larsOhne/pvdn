from tqdm import tqdm
import argparse
import os

from pvdn.bboxes import BoundingBoxDataset
from pvdn.detection.model.proposals import DynamicBlobDetector
from pvdn.metrics.convert import gt_to_results_format
from pvdn.metrics import BoundingBoxEvaluator


if __name__ == "__main__":
    """ Generates the bounding boxes for all sets in"""
    parser = argparse.ArgumentParser(description="Generates the bounding boxes for "
                                                 "all splits (train, val, test).")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="/path/to/dataset/day/")
    parser.add_argument("--yaml", type=str, required=True,
                        help="path to the .yaml configuration file for the blob "
                             "detector specification.")
    args = parser.parse_args()

    # path checks
    if not os.path.isfile(args.yaml):
        raise FileNotFoundError(
            f"The specified config file {args.yaml} does not exist. Please check."
        )
    if not os.path.isdir(args.data_dir):
        raise NotADirectoryError(
            f"The specified data directory {args.data_dir} cannot be found. Please "
            f"check."
        )

    detector = DynamicBlobDetector.from_yaml(args.yaml)

    for split in ("test", "train", "val"):
        print("\n", split)
        ds = BoundingBoxDataset(path=os.path.join(args.data_dir, split),
                                blob_detector=detector)
        ds.generate_bounding_boxes(verbose=True)
        results = gt_to_results_format(ds.bounding_box_path)
        evaluator = BoundingBoxEvaluator(os.path.join(args.data_dir, split))
        evaluator.load_results_from_dict(results)
        performance = evaluator.evaluate(verbose=True)
        print("Performance:", performance)
        del ds
        del evaluator

    print("Finished successfully.")
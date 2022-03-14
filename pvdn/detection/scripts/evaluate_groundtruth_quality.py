from pvdn.metrics.convert import gt_to_results_format
from pvdn.metrics import BoundingBoxEvaluator
from pvdn import BoundingBoxDataset

import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluates the quality of the groundtruth for a given data split."
    )
    parser.add_argument("--data_dir", type=str, help="Path to the data split to be "
                                                     "used.", required=True)
    opts = parser.parse_args()

    for split in ("train", "test", "val"):
        ds = BoundingBoxDataset(path=os.path.join(opts.data_dir, split),
                                blob_detector=None)
        results = gt_to_results_format(ds.bounding_box_path)
        evaluator = BoundingBoxEvaluator(os.path.join(opts.data_dir, split))
        evaluator.load_results_from_dict(results)
        performance = evaluator.evaluate(verbose=True)
        print(f"{split}:", performance)
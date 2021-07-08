from pvdn.metrics.convert import gt_to_results_format
from pvdn.metrics import BoundingBoxEvaluator

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluates the quality of the groundtruth for a given data split."
    )
    parser.add_argument("--data_dir", type=str, help="Path to the data split to be "
                                                     "used.", required=True)
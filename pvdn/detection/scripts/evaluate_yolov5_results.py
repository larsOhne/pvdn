import os
import argparse
from pvdn.metrics.convert import coco_to_results_format
from pvdn.metrics import BoundingBoxEvaluator
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_file", "-y", type=str,
                        help="/path/to/best_predictions.json")
    parser.add_argument("--out_dir", "-o", type=str,
                        help="/path/to/output/directory")
    parser.add_argument("--data_dir", "-d", type=str,
                        help="/path/to/pvdn/dataset/day")
    parser.add_argument("--conf_thresh", "-c", type=float, default=0.5,
                        help="[OPTIONAL] Confidence threshold. Default is 0.5.")
    opts = parser.parse_args()

    os.makedirs(opts.out_dir, exist_ok=True)
    pred_path = os.path.join(opts.out_dir, "predictions.json")
    preds = coco_to_results_format(coco_path=opts.yolo_file)
    with open(pred_path, "w") as f:
        json.dump(preds, f)

    evaluator = BoundingBoxEvaluator(data_dir=opts.data_dir)
    evaluator.load_results_from_dict(results=preds)
    performance = evaluator.evaluate(conf_thresh=0.5, verbose=True)
    perf_path = os.path.join(opts.out_dir, "performance_pvdn_metrics.json")
    with open(perf_path, "w") as f:
        json.dump(performance, f)

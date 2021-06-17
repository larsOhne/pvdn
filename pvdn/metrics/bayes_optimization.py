import argparse
import csv
import os
import numpy.random
from pvdn import BoundingBoxDataset
from pvdn.metrics.bboxes import BoundingBoxEvaluator
import pvdn.metrics.bboxes
import pvdn.metrics.convert
from pvdn.detection.model.proposals import DynamicBlobDetector
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import numpy as np
import pickle


def f(params):
    """
    Calculating metrics for optimization
    :return: 1-box_quality which gets minimized -> int
    """
    nms_distance = 13
    padding = 5
    k, w, dev_thresh, nms_distance, data_path, output_dir, bbox_loss = [params[i][1]
                                                                      for i in range(len(params))]
    # print(k, w, padding, dev_thresh, nms_distance)
    dataset = BoundingBoxDataset(path=data_path,
                                 blob_detector=DynamicBlobDetector(k=k, w=w, padding=padding, dev_thresh=dev_thresh,
                                                                   nms_distance=nms_distance))
    dataset.generate_bounding_boxes()
    out = pvdn.metrics.convert.gt_to_results_format(gt_bbox_dir=data_path + "/labels/bounding_boxes")
    # print(out)
    evaluator = BoundingBoxEvaluator(data_path)
    evaluator.load_results_from_dict(out)
    r = evaluator.evaluate(verbose=False)
    alpha = 0.1
    n_boxes = [len(v["boxes"]) for v in out.values()]
    mean_boxes_per_img = np.mean(n_boxes)
    box_and_f1 = r["f1_score"] * r["box_quality"]
    if bbox_loss:
        loss = 1 - r["box_quality"] + alpha * mean_boxes_per_img
    else:
        loss = 1 - r["box_quality"]
    result = ([k, w, padding, dev_thresh, nms_distance, r["precision"],
               r["recall"], r["f1_score"],
               r["box_quality"], box_and_f1, mean_boxes_per_img, loss])
    results_to_csv(result, output_dir)
    return loss


def optimize(data_path, output_dir, seed, bbox_loss):
    """
    Bayes Optimizer using Tree Parzen Estimator

    :param data_path: Path to dataset
    :param output_dir: Path to output csv-file
    :param seed: Sets seed (int) for reproduction purposes
    :param bbox_loss: flag for enabling bounding box loss
    """

    trials_step = 1  # how many additional trials to do after loading saved trials
    max_trials = 20  # initial max_trials

    try:  # try to load an already saved trials object and increase max_trials
        trials = pickle.load(open(output_dir + "/model.hyperopt", "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
        rstate = None
    except:  # create a new trials object and start searching with given seed
        trials = Trials()
        rstate = np.random.RandomState(seed)

    # Lower and upper bound for each variable of the variable set which gets optimized
    space = [('k', hp.quniform('k', 0.25, 0.75, q=0.05)),
             ('w', hp.choice('w', np.arange(5, 25+1, dtype=int))),
             # ('padding', hp.choice('padding', np.arange(10, 15 + 1, dtype=int))),
             ('dev_thresh', hp.quniform('dev_thresh', 0, 0.1, q=0.01)),
             ('nms_distance', hp.choice('nms_distance', np.arange(1, 9 + 1,
                                                                  dtype=int))),
             ('data_path', data_path),
             ('output_dir', output_dir),
             ('bbox_loss', bbox_loss)]
    best = fmin(fn=f, space=space, algo=tpe.suggest, max_evals=20, rstate=rstate)
    print(f"Optimal value of x: {best}")

    with open(output_dir + "/model.hyperopt", "wb") as fa:
        pickle.dump(trials, fa)
    # loop indefinitely
    while True:
        optimize(data_path, output_dir, None)


def results_to_csv(result, output_dir):
    """ Saves result to csv """
    file_exists = os.path.isfile(output_dir + '/result.csv')
    with open(output_dir + '/result.csv', 'a', ) as csvfile:
        headers = ['k', 'w', 'padding', 'dev_thresh', 'nms_distance', 'precision', 'recall', 'f1_score', 'box_quality',
                   'f1 * box_quality', 'mean_bbox_per_img', 'loss']
        writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="/home/lukas/Development/datasets/PVDN/day/val",
                        help="Path to "
                                                                          "the "
                                                                     "dataset")
    parser.add_argument("--output_dir", type=str, default="optimized", help="Path to "
                                                                      "results")
    parser.add_argument("--seed", type=int, default=4, help="Choose a random seed")
    parser.add_argument("--bbox_loss", action="store_true", help="Flag for enabling "
                                                                 "the bounding box "
                                                                 "loss")
    args = parser.parse_args()
    optimize(args.data_path, args.output_dir, args.seed, args.bbox_loss)

## Optimize BlobDetector Parameters

To optimize the BlobDetector Parameters using Bayesian Optimization
with [hyperopt](https://github.com/hyperopt/hyperopt) with
the [Tree of Parzen Estimator (TPE)](https://papers.nips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf)
run:

```
python3 bayes_optimization.py --data_path "path/to/data" --output_dir "path/to/output_dir" --seed # set random seed (int)
```

This will automatically optimize the parameters on the given dataset to the best Bounding Box Quality which is defined
in [bboxes.py](bboxes.py). To set the bounds of the variables, change the search space
in [bayes_optimization.py](bayes_optimization.py). The process will run indefinitely until stopped and writes the
hyperopt progress to a Trials File which gets loaded the next time you start the process.

You can optionally enable the incentivation of producing only as few bounding boxes 
as possible via the `--bbox_loss` flag.

The results will be saved in a csv-file which contains the value of the variables with additional evaluation metrics
such as f1_score and box quality
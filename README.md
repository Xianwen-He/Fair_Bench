# Fair Bench

We will provide a more detialed README soon. For now, please refer to the uploaded csv files for our experimental results. If you are interested in reproduce our results, please take a look at the following commands.


## DEMO

To run a quick demo for LabelDebias, please run the following command:

```bash
cd label_debias
python run_label_debias_Linear.py --metric DP \
                           --model LogReg \
                           --max_iteration 1000 \
                           --n_iters 10 \
                           --save_folder ./results/LogReg \
                           --save_file LogReg_DP
```


To run a quick demo for Fair Constraints, please run the following command:

```bash
cd fair_constraints
python run_disparate_impact.py  --model LogReg \
                                --save_file constraints
```


To run a quick demo for LAFTR, please run the following command.


First, generate the fair intermediate representations:
```bash
cd laftr
python run_laftr.py conf/transfer/laftr_then_naive.json -o exp_name="laftr_new/adult",train.n_epochs=1000 --data adult --dirs local  # train a LAFTR model on the Adult dataset
```

Then, train a classifier on the learned representations:
```
python run_repr.py  --experiment_folder ./experiments/laftr_new/adult \
                    --model LogReg \
                    --save Adult_transfer_logreg
```


To run a quick demo for thresholding, please run the following command:
```bash
cd thresholding
python run_thresholding.py --model LogReg \
                           --save_file thresholding \
                           --max_iteration 1000
```


## References

The data comes from the following resources:

COMPAS: https://github.com/propublica/compas-analysis

Adult: https://archive.ics.uci.edu/dataset/2/adult


This repository is built upon the following repositories:

LabelDebias: https://github.com/google-research/google-research/tree/master/label_bias

Fair Constraints: https://github.com/mbilalzafar/fair-classification

LAFTR: https://github.com/VectorInstitute/laftr

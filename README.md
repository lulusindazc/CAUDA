# CA-UDA
This is the Pytorch implementation for our paper **CA-UDA: Class-Aware Unsupervised Domain Adaptation with Optimal Assignment and Pseudo-Label Refinement**

## Requirements
- Python 3.7
- Pytorch 1.1
- PyYAML 5.1.1

## Dataset 
The structure of the dataset should be like

```
Office-31
|_ category.txt
|_ amazon
|  |_ back_pack
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ bike
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ ...
|_ dslr
|  |_ back_pack
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ bike
|     |_ <im-1-name>.jpg
|     |_ ...
|     |_ <im-N-name>.jpg
|  |_ ...
|_ ...
```
The "category.txt" contains the names of all the categories, which is like
```
back_pack
bike
bike_helmet
...
```

We mainly do experiments on four datasets: Office-31, VisDA-2017, Digit-Five and ImageCLEF-DA.
## Training
```
sh train.sh ${config_yaml} ${adaptation_method} ${experiment_name}
```
or 
```
python train_debug.py --cfg ${config_yaml} --method ${adaptation_method} --exp_name ${experiment_name}

```
Dataset can be chosen by using different config files.

For example, for the Office-31 dataset,
```
python train_debug.py --cfg experiments/config/Office-31/C2C/office31_train_amazon2dslr_cfg.yaml --method CENTER_KMEANS_SP --exp_name office31_a2d
```
for the VisDA-2017 dataset,
```
python train_debug.py --cfg experiments/config/VisDA-2017/CAN/visda17_train_train2val_cfg.yaml --method CENTER_KMEANS_SP --exp_name visda17_train2val
```
The experiment log file and the saved checkpoints will be stored at ./experiments/ckpt/${experiment_name}

## Test

```
python test.py --cfg ${config_yaml} --exp_name ${experiment_name}
```
Example: 
```
python test.py --cfg experiments/config/Office-31/office31_test_amazon_cfg.yaml --exp_name visda17_test
```


## Contact
If you have any questions, please contact me via zhangcan.lulu@gmail.com.

## Thanks to third party
The way of setting configurations is inspired by <https://github.com/rbgirshick/py-faster-rcnn>.


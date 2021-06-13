# DSAI_FinalProject-CouponPurchasePrediction
Final Project of NCKU DSAI Course: Coupon Purchase Prediction

Environment: Windows 10 + Anaconda 3

## Environment Setting
1. $ conda env create --name <env_name> -f requirement.yml
2. $ conda activate <env_name>

## Run the program
[usage] 
$ python data.py --re_train False --train train.csv --sample_number 100000 --re_test False --test test.csv
Argument List:
* --re_train: Generate a new train file generated randomly. (True of False. default: False)
* --train: if "--re_train" is needed, assign a output path for the training file.(path/to/train, default: "train.csv")
* --sample_number: Assign a positive integer(MAX: 2833180) of sampling for training.(Default: 100000)
* --re_test: Generate a test file for testing.(True or False. Default: False)
* --test: If "--re_test" is needed, assign a output path for the training file.(path/to/test, default: "test.csv")

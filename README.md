# DSAI_FinalProject-CouponPurchasePrediction
Final Project of NCKU DSAI Course: Coupon Purchase Prediction

OS: Windows 10

CPU: Intel i7-9700K

Python version: 3.8.8

Please download the data.zip from the google drive's link below and unarchieve the data folder under the main folder.

[Google Drive download link](https://drive.google.com/file/d/1qBxp2qyWMJFQ6AFF3c5ZMSsUZzxkH2KF/view?usp=sharing) 
## Environment Setting
```
$ pip install -r requirement.txt
```
## Run the program
[usage] 
```
$ python data.py
```
(The execution of data.py is optional because the required data has been generated in repo and it takes about 45 minutes to finish.)
```
$ python main.py --model rfc --output submission.csv
```
Argument List:
* --option: Choose the ML method from RandomForest(rfc) or LightGBM(lgb). (default: rfc)
* --output: Specify the output (path/to/submission, default: submission.csv)


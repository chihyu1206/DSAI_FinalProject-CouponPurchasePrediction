# DSAI_FinalProject-CouponPurchasePrediction
Final Project of NCKU DSAI Course: Coupon Purchase Prediction

OS: Windows 10
CPU: Intel i7-9700K
Python version: 3.8.8

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
$ python main.py --option predict --output submission.csv
```
Argument List:
* --option: Fit the Random Forest Classifier model again and save it (retrain) or Predict the result by saved model directly (predict). (default: predict)
* --output: Specify the output (path/to/test, default: "test.csv")

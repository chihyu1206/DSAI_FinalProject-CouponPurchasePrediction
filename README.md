# DSAI_FinalProject-CouponPurchasePrediction
Final Project of NCKU DSAI Course: Coupon Purchase Prediction

OS: Windows 10

CPU: Intel i7-9700K

Python version: 3.8.8

Please download the data.zip from the google drive's link below and unarchieve the data folder under the main folder.

[Google Drive download link](https://drive.google.com/file/d/19uqDb53Mo1mdgefnp24-GfvVTPTG-Z80/view?usp=sharing)

[期末Project詳細報告](https://docs.google.com/document/d/1RT6mosSeknuJ0tAtgALxZS4BoPC5sGvSbj0iDR5L0bM/edit?usp=sharing)

[期末Project口頭報告(6/22報告用投影片)]()
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

## Result
### Random Forest
Ranking in top 9% (107 / 1072 teams)
![image](https://github.com/chihyu1206/DSAI_FinalProject-CouponPurchasePrediction/blob/main/Result/RandomForest.jpg)
![image](https://github.com/chihyu1206/DSAI_FinalProject-CouponPurchasePrediction/blob/main/Result/RandomForestRanking.jpg)

### LightGBM
Ranking in 48% (521 / 1072 teams)
![image](https://github.com/chihyu1206/DSAI_FinalProject-CouponPurchasePrediction/blob/main/Result/LightGBM.jpg)
![image](https://github.com/chihyu1206/DSAI_FinalProject-CouponPurchasePrediction/blob/main/Result/LightGBMranking.jpg)

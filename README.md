# MedicalCost_Prediction
 
The purpose of this project was to check for correlation and prediction possibility for a medical dataset. 

The dataset used is this one: https://www.kaggle.com/datasets/waqi786/medical-costs

"This dataset contains detailed information about medical costs for individuals over the period from 2010 to 2020. It includes various attributes such as age, sex, BMI, number of children, smoking status, and region. These attributes are essential in understanding the factors that influence medical costs and can be used for predictive modeling, statistical analysis, and research purposes."

First, I wanted to analyze the data to check for distribution and split of the different features. 

After that, I used my GPU alongside XGBRegression model to train and test a model which later would be used for fitting. Initial results seemed promising: 

Mean Absolute Error: 255.77263634114584\
Mean Squared Error: 87840.25774818643\
R-squared: 0.9976068143268261

![alt text](Images/ActualvsPredicted.png)

After this, I moved over from the Jupyter file to two scripts: 

train_model.py\
predict_cost.py\

The role of these two files was to train the model and then predict the cost with as little configuration as possible. The predict_cost script also included user input fields, which would later assist us when creating a flask back-end with HTML front-end. 

In app.py and index.html you can see the flask back-end that is connected to the front end. 

The web-server can be started by typing "python -m http.server" into the terminal

The end result looks like this: 

![alt text](Images/FrontEnd.png)

However, I was not happy due to the fact that there seemed to be a large bias against smokers, and while it's logical, I wanted to see if I did any mistakes during the model training. So I created the balanced version. Additionally, the model seemed to be very biased to numbers between 6k and 8k and 16k to 20k. Nothing in between. 

Well, this was easy to solve by a simple distribution chart: 

![alt text](Images/DistributionChart.png)



![alt text](image.png)
# Energy Consumption Prediction

### Project Description
As the years go by, our dependence on energy continues to grow, driven by population expansion, technological advancements, and urbanization. Energy has become an essential part of modern life, powering everything from homes and industries to transportation systems and communication networks. However, this increasing energy consumption raises significant concerns about its long-term sustainability, especially for natural environments. <br><br>
This project is about how to determine energy consumption prediction which you can get the dataset from [Kaggle](https://www.kaggle.com/datasets/arnavsmayan/smart-home-energy-usage-dataset). There are 1.000.000 data records from this dataset and it consists of several columns including...
- Timestamp, which contains date and time of recorded data
- Home id
- Energy consumption (kWh)
- Temperature settings (C)
- Occupancy status
- Appliances (e.g. HVAC, Lighting, Refrigerator, Electronics, Dishwasher, and Washing Machine)
- Usage duration minutes
- Season
- Day of Week
- Holiday

The target of this prediction is energy consumption column meanwhile other columns are features

**Note:** Seasons data are not correlated with the month and day from timestamp column. In one day, there are 4 seasons recorded but the differences between those records are from hours

### Models
The models that I used for this project are:
- Decision Tree Regression
- Random Forest Regression
- LightGBM

### How to Install the Dependencies
These steps must be doing sequentially <br>
1. Make virtual environment by using pipenv, to do this...
    ```
    pip install pipenv
    ```
2. Install the libraries by using pipenv
    ```
    pipenv install numpy scikit-learn==1.5.2 flask
    ```
3. Install a library which creates WSGI server. Because this project made in Windows sytem, so I used waitress library
    ```
    pipenv install waitress
    ```

### Containerization
The python version which I used for this project is 3.12.4 so the Python image which I was going to use is ```3.11.10-slim``` 

1. Run Python image
    ```
    docker run -it --rm python:3.11.10-slim
    ```
2. Access the container's terminal
    ```
    docker run -it --rm --entrypoint=bash python:3.11.10-slim
    ```
3. Create a dockerfile. You can check on my Dockerfile from this repository
4. Build Docker container
    ```
    docker build -t midterm-project .
    ```
5. Run the container
    ```
    docker run -it --rm -p 9696:9696 midterm-project
    ```
You will see that the deployment server is on running so if you run ``predict-test.py``, it will run perfectly. You can run this by command line or run it in VSCode
```
python predict-test.py
```

### Fun Fact
If you want to run the notebook, it takes so much times to run it approximately more than 2 hours (especially in Random Forest Regression part) because this dataset is so big. This was also a consideration why I didn't use XGBoost for model 
# Disaster Response Pipeline Project
## Summary
In this repository, you'll find a data set containing real messages that were sent during disaster events. You also find a machine learning pipeline to categorize these events.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Files

- `process_data.py` contains the __ETL Pipeline__ for loading and cleaning data
- `train_classifier.py` contains the __ML Pipeline__ for training a new classifier.

### Folders

- Folder '`./App`' contains the web application for visualization.
- Folder '`./Data`' contains the csv. files and the sqlite database.
- Folder '`./Model`' contains the trained model of the classifier.

## Instructions

> __Remark__  
In the following section there will be a short instruction provided how to interact with the project.

1. Run the following commands in the project's root directory to set up your database and model.

    1. To run ETL pipeline that cleans data and stores in database             
        ```            
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```
    1. To run ML pipeline that trains classifier and saves
        ``` 
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```

2. Run the following command in the app's directory to run your web app.
    ```
    python run.py
    ```

3. Go to http://0.0.0.0:3001/

![](./app/app.png)

# Acknowledgements
Thanks to Figure Eight for providing the .csv files in order to analyze such real word problem's scenarios.

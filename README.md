# Disaster Response Pipeline Project

## Motivation:

Apply software and data engineering skills to analyze disaster data to build a model for an API that classifies disaster messages. Using the web app an emergency worker can input a new message and get classification result in several categories such as "water", "fire", "food", etc. Such classification can help emergency workers notify an appropriate disaster relief agency.

## Data:

The dataset was provided by Figure Eight and contained real messages that were sent during disaster events.

## Final Result:

Web app that prompts users to enter a message and returns corresponding category predictions based on the ML pipeline.

## Files in the repository:

app<br>
| - templates <br>
| |- master.html # main page of web app <br>
| |- go.html # classification result page of web app <br>
| |- about.html # about page of the project <br>
|- run.py # Flask file that runs app <br>
data <br>
|- disaster_categories.csv # data to process <br>
|- disaster_messages.csv # data to process <br>
|- process_data.py <br>
|- DisasterResponse.db # database to save clean data to <br>
|- ETL Pipeline Preparation.ipynb <br>
models <br>
|- train_classifier.py <br>
|- ML Pipeline Preparation (1).ipynb <br>
|- classifier.pkl # saved model <br>
README.md <br>

## Instructions:

1. Install dependencies in your virtual environment by running:

   `conda create --name <env> --file <requirements.txt>`

2. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

     Note: initial model training may take a few hours depending on your machine.

3. Run the following command in the app's directory to run your web app.
   `python run.py`

4. Go to http://0.0.0.0:3001/

## Web application screenshots:

![main](data/img/main.png)
![main](data/img/genres.png)
![main](data/img/categories.png)
![main](data/img/corr.png)

After entering the following message: "I would like to know if the earthquake is over."

![main](data/img/classification1.png)
![main](data/img/classification2.png)

![main](data/img/about.png)

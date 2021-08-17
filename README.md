# Disaster Response Pipeline Project

## Motivation:

---

Apply software and data engineering skills to analyze disaster data to build a model for an API that classifies disaster messages. Using the web app an emergency worker can input a new message and get classification result in several categories such as "water", "fire", "food", etc. Such classification can help emergency workers notify an appropriate disaster relief agency.

## Classifier Description:
- CountVectorizer
- TfidfTransformer
- AdaBoostClassifier
- GridSearchCV

#### Class Imbalance
![main](data/img/categories.png)
As you can see on the histogram above, we have a significant class imbalance. Since this is a Multilabel multiclass classification problem, we can't simply apply SMOTE to balance out classes. Therefore, I decided to calculate an oversampling ratio for each class. Then, for each message in the training set, I check the categories it belongs to and their corresponding oversampling ratios, keeping track of the **maximum one = r.** I then duplicate this message **r** number of times. 

## Files in the repository:

---

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
README.md <br>
requirements.txt #list of dependencies

#### Note:
The classifer.pkl is not uploaded due to the large size of the file. The initial run of the train_classifier.py will create a model in the specified directory. 
Also note, that it might take a while due to cross validation.

## Instructions:

---

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

## Result:

---

Web app that prompts users to enter a message and returns corresponding category predictions based on the ML pipeline.

| Category | precision | recall | f1-score | support |
| ---------------------- | -----| ------ | -------- | ------- |
|                related | 0.76 | 1.00 | 0.86 | 4011 |
|                request | 0.81 | 0.19 | 0.30 | 896 |
|                  offer | 0.00 | 0.00 | 0.00 | 30 |
|            aid_related | 0.42 | 1.00 | 0.59 | 2206 |
|           medical_help | 0.72 | 0.04 | 0.08 | 437 |
|       medical_products | 0.79 | 0.10 | 0.18 | 304 |
|      search_and_rescue | 0.79 | 0.17 | 0.28 | 152 |
|               security | 0.00 | 0.00 | 0.00 | 92 |
|               military | 0.00 | 0.00 | 0.00 | 199 |
|                  water | 0.63 | 0.75 | 0.68 | 346 |
|                   food | 0.77 | 0.66 | 0.71 | 579 |
|                shelter | 0.85 | 0.32 | 0.46 | 476 |
|               clothing | 0.74 | 0.51 | 0.60 | 73 |
|                  money | 1.00 | 0.02 | 0.04 | 110 |
|         missing_people | 0.63 | 0.30 | 0.41 | 63 |
|               refugees | 0.86 | 0.03 | 0.06 | 182 |
|                  death | 1.00 | 0.03 | 0.05 | 252 |
|              other_aid | 1.00 | 0.00 | 0.00 | 687 |
| infrastructure_related | 0.55 | 0.02 | 0.03 | 335 |
|              transport | 0.59 | 0.25 | 0.35 | 273 |
|              buildings | 0.57 | 0.12 | 0.20 | 250 |
|            electricity | 0.70 | 0.06 | 0.11 | 118 |
|                  tools | 0.00 | 0.00 | 0.00 | 34 |
|              hospitals | 0.29 | 0.03 | 0.06 | 62 |
|                  shops | 0.00 | 0.00 | 0.00 | 25 |
|            aid_centers | 0.00 | 0.00 | 0.00 | 58 |
|   other_infrastructure | 0.00 | 0.00 | 0.00 | 213 |
|        weather_related | 0.91 | 0.48 | 0.62 | 1509 |
|                 floods | 0.92 | 0.42 | 0.58 | 460 |
|                  storm | 0.87 | 0.08 | 0.15 | 487 |
|                   fire | 0.35 | 0.27 | 0.31 | 44 |
|             earthquake | 0.90 | 0.79 | 0.84 | 528 |
|                   cold | 0.83 | 0.14 | 0.24 | 108 |
|          other_weather | 0.14 | 0.00 | 0.01 | 281 |
|          direct_report | 0.81 | 0.12 | 0.21 | 1029 |
|            no_category | 0.00 | 0.00 | 0.00 | 1267 |
|              micro avg | 0.64 | 0.49 | 0.56 | 18176 |
|              macro avg | 0.56 | 0.22 | 0.25 | 18176 |
|           weighted avg | 0.66 | 0.49 | 0.46 | 18176 |
|            samples avg | 0.59 | 0.50 | 0.48 | 18176 |

## Licensing and Acknowledgements:

---

The dataset was provided by Figure Eight and contained real messages that were sent during disaster events.

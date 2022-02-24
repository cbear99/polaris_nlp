To run the code for this implementation, first enter the python venv using source venv/bin/activate. 
Then, run train regression using ./train_regression.py cases_training.csv stopword. If you want to use a larger dataset than cases_training, ensure that it contains
an additional column for followed_up_for_trafficking. 
This will generate the regression. Finally run predict cases using ./predict_cases.py 'cases.csv' 'mapping.csv' stopwords. This will output a ranking of the cases
in mapping.csv

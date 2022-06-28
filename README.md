This is a simple regression problem, solved using RandomForestRegressor

Before using the module of this package make sure unimportant columns are removed from the data like id,date,zipcode etc and your data has no missing values.

## To excute the script
First install the dependencies specified in env.yml file.

Now open cmd prompt and start mlflow server by using below command
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5000

mlflow server starts and saves the mlflow log files whaterver you want to in "./artifacts" folder.
you can check whether it is working or not in http://127.0.0.1:5000/

Now open the cmd prompt and type the below command to run test file which will run ingest_data.py, train.py, and score.py. The below command will take the dataset and split it as train and validation datasets and then train the model and also provide the test scores.

**python test_model.py**

If you want to run individual file then follow the below steps

1. To get the data and create training and validation datasets then save those datasets in given path.
**python ingest_data.py < path to save ouput files >**
Example: python ingest_data.py C:\Users\saikumar.godha\Downloads\basic_regg_lib\output_folder

2. The train the data by giving datasets path then saves the model in pickle formate in given path.
**python train.py < directory path where ouput files are stored >**
Example: python train.py C:\Users\saikumar.godha\Downloads\basic_regg_lib\output_folder

3. To show the score for the model by taking model path and datasets path
**python score.py < model.pickle path > < datasets path >**
Example: python score.py C:\Users\saikumar.godha\Downloads\basic_regg_lib\output_folder\model.pickle C:\Users\saikumar.godha\Downloads\basic_regg_lib\output_folder

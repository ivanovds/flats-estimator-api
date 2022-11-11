# Flat price estimator API 

This is a simple Machine Learning model that estimates the price of a flat in Kyiv.
It was learned on data from [dom.ria](https://dom.ria.com/) as of 2020 year.
To interact with the model, you can use the REST API.

## Technology stack
* Python 3, Scikit-learn, requests
* Django, DRF

## Model training
* Used 2 algorithms: Random Forest and Gradient Boosting
* Main six features: center_distance, metro_distance, azimuth, area, number of floors, floor
* The model was trained on 80% of the data, and the remaining 20% was used for validation
* Quality criteria: 
1) Mean Absolute Percentage Error (MAPE) = 12.2%
2) MEDium Absolute Percentage Error (MedAPE) = 4.2%
3) Coefficient of determination (R^2) = 0.84

* Code for model training is in the [kyiv_flats_valuation.py](kyiv_flats_valuation.py) file
* Detail information about this project (in Ukrainian language) is in [Ivanov_bakalavr.pdf](staticfiles/Ivanov_bakalavr.pdf) file 


## Usage
API root view:
![step1](staticfiles/readme/Screenshot_1.png?raw=true "Title")

List of users:
![step2](staticfiles/readme/Screenshot_2.png?raw=true "Title")

List of your estimated flats (authentication required):
![step3](staticfiles/readme/Screenshot_3.png?raw=true "Title")

Login page:
![step4](staticfiles/readme/Screenshot_4.png?raw=true "Title")

List of your estimated flats:
![step4](staticfiles/readme/Screenshot_5.png?raw=true "Title")

List of your estimated flats (here you can create a new one):
![step4](staticfiles/readme/Screenshot_8.png?raw=true "Title")

Detail view of single flat (here you can see the price estimation):
![step4](staticfiles/readme/Screenshot_6.png?raw=true "Title")

Detail view of single flat (here you can update it):
![step4](staticfiles/readme/Screenshot_7.png?raw=true "Title")

## Installation

* Clone repository:
```bash
git clone https://github.com/ivanovds/flats-estimator-api
```

* Go to the project folder:
```bash
cd project-container
```

* Create environment variables file:
```bash
cp project/api/env.example project/api/.env
```

* Create a virtual environment:

On macOS and Linux:
```bash
python3 -m venv venv
```
On Windows:
```bash
py -m venv venv
```

* Switch your virtal environment in the terminal:

On macOS and Linux:
```bash
source venv/bin/activate
```
On Windows:
```bash
venv\Scripts\activate
```

* Then install all packages you need.

All requirements are stored in requirements.txt.
Use the package manager [pip](https://pip.pypa.io/en/stable/) 
to install by command:

```bash
pip install -r requirements.txt
```

* Add Django Server in Run/Debug Configuration.

* Now you can run your Django Server by command:
```bash
py manage.py runserver
```
and visit http://127.0.0.1:8000/api/v1/


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[GNU](https://choosealicense.com/licenses/gpl-3.0/)
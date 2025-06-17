<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Unlicense License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- ABOUT THE PROJECT -->
## About The Project

This project is a simple **machine learning pipeline** that trains a model to predict the price range of mobile phones using structured input features. It demonstrates:

- How to automate ML training using GitHub Actions
- Committing trained models directly to the repo
- Tracking performance metrics like accuracy and regression error
- Modular ML code structure with versioned artifacts

You can trigger retraining by updating the dataset (`data/raw/train.csv`) or running the GitHub Actions workflow manually.


<!-- BUILT WITH -->
## Built With

* [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
* [scikit-learn](https://scikit-learn.org/)
* [GitHub Actions](https://github.com/features/actions)


<!-- GETTING STARTED -->
## Getting Started

These instructions will help you set up the project locally and run it manually. GitHub Actions will automatically retrain the model whenever you update `train.csv`.


<!-- INSTALLATION -->
### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/darrellathaya/phone-predictor
   cd phone-predictor

2. Install missing dependencies
   ```sh
   pip install -r requirements.txt
   

<!-- PREREQUISITES -->
### Prerequisites
1. Register & Login to SonarQube Cloud
   ```sh
   a. Create a new organization
   b. Create a new project by linking it to your Github Repository
   c. Create a new token 
      Head to My Account > Security > Generate Tokens, label it as SONAR_TOKEN

2. Github
   ```sh
   a. Create a new repository
   b. Add the Github Secrets
      Head to Settings > Secrets & Variables > Actions, Click on "New repository secret" as SONAR_TOKEN 


<!-- USAGE EXAMPLES -->
## Usage

1. Train the model locally
   ```sh
   python src/model.py
   ```

2. Running the web app locally
   ```sh
   uvicorn app.main:app --reload
   ```

3. Run via Railway
   ```sh
   a. Add new data into train.csv

   b. Push the changes into your Github Repository


<!-- DIRECTORY -->
## Project Directory
```sh
.
├── app/
│   ├── _pycache_/
│   │   └── main.cpython-313.pyc
│   ├── main.py
│   └── main.test.py
├── data/
│   └── raw/
│       └── train.csv
├── mlartifacts
├── mlruns
├── models/
│   ├── accuracy.txt
│   ├── label_encode.json
│   ├── meta.json
│   └── price_range_model.pkl
├── src/
│   ├── train_model.py
│   └── train_model.test.py
├── templates/
│   ├── static/
│   │   └── style.css
│   └── index.html
├── Dockerfile
├── requirements.txt
└── sonar-project.properties
```

<!-- Known Issues and Fixes -->
## Known Issues and Fixes
MLFLOW

![Screenshot 2025-06-17 183811](https://github.com/user-attachments/assets/2b0e9155-82a9-456d-bc71-f8e52f17e4f4)

   the issues above occured because the neccessary requirements was not installed. This issue can be fixed by adding the text bellow to the requirements.txt
   ```sh
mlflow
   ```

![Screenshot 2025-06-17 183027](https://github.com/user-attachments/assets/6f29d148-164e-487e-81a9-129d9f613356)

   the issues above occured because the the correct experiment was not defined properly. It was resolved by adding the following code
   ```sh
mlflow.set_experiment("PhonePricePrediction")
   ```


NEW MODEL IMPLEMENTATION
![image](https://github.com/user-attachments/assets/245ee94f-1068-47bd-bee6-d683f4e5b8de)

   the issues above occured because the neccessary requirements was not installed. This issue can be fixed by adding the text bellow to the requirements.txt
   ```sh
xgboost
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/darrellathaya/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/darrellathaya/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/darrellathaya/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/darrellathaya/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/darrellathaya/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/darrellathaya/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/darrellathaya/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/darrellathaya/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/darrellathaya/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/darrellathaya/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/darrellathaya
[product-screenshot]: images/screenshot.png

[Java.io]: https://img.shields.io/badge/Java-ED8B00?style=for-the-badge&logo=openjdk&logoColor=white
[Java-url]: https://www.java.com/

[MsgPack.io]: https://img.shields.io/badge/MessagePack-000000?style=for-the-badge&logo=data&logoColor=white
[MsgPack-url]: https://msgpack.org/

[Jackson.io]: https://img.shields.io/badge/Jackson-2F3134?style=for-the-badge&logo=code&logoColor=white
[Jackson-url]: https://github.com/FasterXML/jackson

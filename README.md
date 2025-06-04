<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Unlicense License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- ABOUT THE PROJECT -->
## 📦 About The Project

This project is a simple **machine learning pipeline** that trains a model to predict the price range of mobile phones using structured input features. It demonstrates:

- How to automate ML training using GitHub Actions
- Committing trained models directly to the repo
- Tracking performance metrics like accuracy and regression error
- Modular ML code structure with versioned artifacts

You can trigger retraining by updating the dataset (`data/raw/train.csv`) or running the GitHub Actions workflow manually.


<!-- BUILT WITH -->
## 🛠️ Built With

* [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
* [scikit-learn](https://scikit-learn.org/)
* [GitHub Actions](https://github.com/features/actions)


<!-- GETTING STARTED -->
## 🚀 Getting Started

These instructions will help you set up the project locally and run it manually. GitHub Actions will automatically retrain the model whenever you update `train.csv`.


<!-- INSTALLATION -->
### 📦 Installation

1. Clone the repo
   ```sh
   git clone https://github.com/darrellathaya/phone-predictor.git
   cd phone-predictor
   

<!-- USAGE EXAMPLES -->
## Usage

1. Running the web app locally
   ```sh
   uvicorn app.main:app --reload
   ```

2. Train the model locally
   ```sh
   python src/model.py
   ```

3. Run via Github Actions
   ```sh
   a. Add new data into train.csv

   b. Push the changes into your Github Repository


<!-- DIRECTORY -->
## Project Directory
```sh
.
├── app/
│   ├── _pycache_/
│   │   ├── main.cpython-312.pyc
│   │   └── main.cpython-313.pyc
│   └── main.py
├── data/
│   └── raw/
│       └── train.csv
├── models/
│   ├── accuracy.txt
│   ├── chipset_encoder.pkl
│   ├── meta.json
│   ├── model.pkl
│   └── regression_metrics.txt
├── src/
├── _pycache_/
│   │   ├── main.cpython-312.pyc
│   ├── model.py
│   ├── train.py
├── templates/
│   └── index.html
├── Dockerfile
├── README.md
└── requirements.txt
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

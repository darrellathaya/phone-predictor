<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Unlicense License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- ABOUT THE PROJECT -->
## About The Project

This project is a simulation of a simple Java-based tweet storage system, using a multifile approach with `MessagePack` storage format for hot data and `JSON Lines (.jsonl)` for cold data.

The main goals of this project are to demonstrate:
- How to store and read tweets in an efficient format
- Managing hot and cold data based on date
- Implementing schema evolution (add columns, rename, change types, delete columns)
- Simulating basic social media such as timeline and follow

<!-- BUILD -->
## Built With

* [![Java][Java.io]][Java-url]
* [![MessagePack][MsgPack.io]][MsgPack-url]
* [![Jackson][Jackson.io]][Jackson-url]


<!-- GETTING STARTED -->
## Getting Started
This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.


<!-- INSTALLATION -->
### Installation
1. 
   
<!-- USAGE EXAMPLES -->
## Usage

1. Running the web app
   ```sh
   uvicorn main:app.main --reload
   ```

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
│       └── data.csv
├── models/
│   ├── accuracy.txt
│   ├── chipset_encoder.pkl
│   ├── meta.json
│   ├── price_range_model.pkl
│   └── regression_metrics.txt
├── src/
│   └── train_model.py
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
[MsgPack.io]: https://img.shields.io/badge/MessagePack-000000?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciICB3aWR0aD0iMjQiIGhlaWdodD0iMjQiPjxwYXRoIGQ9Ik0xMiAyYTkgOSAwIDEgMCAwIDE4IDkgOSAwIDAgMCAwLTE4em0xIDEzSDExdi0yaDJ2MnptMC00SDExVjZoMnY1eiIvPjwvc3ZnPg==
[Jackson.io]: https://img.shields.io/badge/Jackson-2F3134?style=for-the-badge&logo=jackson&logoColor=white
[Java-url]: https://www.java.com/
[MsgPack-url]: https://msgpack.org/
[Jackson-url]: https://github.com/FasterXML/jackson

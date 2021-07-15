<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
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


<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h3 align="center">Brazilian Name Generator</h3>

  <p align="center">
    Create cool and awkward names with Language Models!
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#build-with">Built With</a></li>
    </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#local-installation">Local Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
      <ul>
        <li><a href="#generate-new-names">Generate New Names</a></li>
        <li><a href="#reproduce-training">Reproduce Training</a></li>
        <li><a href="#docker">Docker</a></li>
      </ul>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

```
Predicted the name: RUNNATAIDENILOS
Prefix: RU
Context Size: 2
Seed: 1
```


Language Models are tasked with assigning a probability to a word or even a sentence.
They correct the misspelled words you type on your cell phone, as well as help your 
personal assistant to understand you.

In this fun project, I used them to make a probabilistic model of the 
characters of Brazilian names using data from the 
[2010 census](https://brasil.io/dataset/genero-nomes/nomes/). Then, I used these
models to generate new names. 

It works by guessing next letters based on the previous ones. For instance, 
what is the most probable name given that the name starts with *Pau...*? For the
English language it will probably be *Paul*, while for Portuguese it will be *Paulo*.
However, if we use a small enough context size (e.g., number of previous letters to 
infer the next one), awkward and cool names start to appear =)

<!-- ABOUT THE PROJECT -->
## Built With

* [Cookiecutter Data Science Project Structure](https://drivendata.github.io/cookiecutter-data-science/)
* Python Data Science Tools (Pandas, Numpy, etc)

<!-- GETTING STARTED -->
## Getting Started

You can use this project with docker or install locally in your machine

### Prerequisites

* Docker 

or

* Linux/WSL
* conda

### Local Installation

1. Clone the repo
    ```sh
    git clone https://github.com/renan-cunha/NameGeneratorBR
    cd NameGeneratorBR/
    ```
2. Create environment
    ```
    make create_environment
    conda activate NameGeneratorBR
    ```
3. Install requirmeents
    ```
    make requirements
    ```

<!-- USAGE EXAMPLES -->
## Usage

The repo has five trained models, from context size equal to 0 (e.g., the next letter
is predicted by how much it appears in the dataset) to 4 (e.g., the previous four letters are used to infer the next one).

### Generate New Names

If you want just to generate a new name, use the ```src/models/predict_model.py``` 
with the following options:

```
Usage: predict_model.py [OPTIONS]

Options:
  -cs, --context_size INTEGER  How much context to use for the language model,
                               The pre-trained models go from 0 to 4
  -p, --prefix TEXT            The beginning of the name to be predicted (OPTIONAL)
  -s, --seed INTEGER           Seed to reproduce experiments (OPTIONAL)
  --help                       Show this message and exit.
```
Ex:
```
(NameGeneratorBR) renan@DESKTOP-AD25DOI:~/git/NameGeneratorBR$ python src/models/predict_model.py -cs 4 -p pau -s 0
Predicted the name: PAULO
Prefix: PAU
Context Size: 4
Seed: 0
```

### Reproduce Training

To reproduce the training, use the command below

````
```make train_model```
````

### Docker

Pull the image

```
docker pull renancunha97/name-generator-br
```

And make new names

```
renan@DESKTOP-AD25DOI:~$ docker run renancunha97/name-generator-br -cs 4 -p pau -s 0
Predicted the name: PAULO
Prefix: PAU
Context Size: 4
Seed: 0  
```


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Renan Cunha - renancunhafonseca@gmail.com

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

If you are curious about Language Models and Natural Language Processing in general, I highly recommend
Jurafsky's drafts of [Speech and Language Processing 3rd edition](https://web.stanford.edu/~jurafsky/slp3/) and his [classes](https://www.youtube.com/channel/UC_48v322owNVtORXuMeRmpA).

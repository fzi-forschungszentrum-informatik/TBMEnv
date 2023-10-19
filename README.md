# TBMEnv - An Environment to Assess the Accuracy of Thermal Building Models under Realistic Conditions

TBMEnv is an program that can be used to evaluate and benchmark models of model-based thermal building optimization algorithms, which are most likely model-based reinforcement learning approaches. The main goal of the environment is to support the evaluation of thermal (machine learning) building models under realistic conditions as this cannot be achieved using available environments like [BOPTEST](https://github.com/ibpsa/project1-boptest) or [Sinergym](https://github.com/ugr-sail/sinergym). For more details and background please see the accompanying [paper](./paper.pdf).



## Installation

Begin by checking out the source code:

```bash
git clone https://github.com/fzi-forschungszentrum-informatik/tbmenv.git
cd tbmenv
```

This is code is written in Python. Given that Python and pip are installed, you can install the environment by typing:

```python
pip3 install -e ./source/
```

As an alternative this repository contains a [docker-compose.yml](docker-compose.yml) specifying a docker runtime. Assuming that [docker](https://docs.docker.com/get-started/) and [docker compose](https://docs.docker.com/compose/gettingstarted/) are available it is possible to start the container with the following command:

```bash 
docker compose up --build
```

The second option may be more suitable, especially for those just wanting to play around with the demo [notebooks](./notebooks/) provided.



## Usage

Please see the [usage example notebook](./notebooks/usage_example.ipynb).



## Building Data

We would like to thank the [Institute for Automation and Applied Informatics](https://www.iai.kit.edu/english/index.php) of the [Karlsruhe Institute of Technology](https://www.kit.edu/english/index.php) for providing the data and information about the building that has been used as blueprint for Scenario 1. The dataset has been recorded in the research project "Flexkälte – Flexibilisierung vorhandener Kälteanlagen und deren optimierter Einsatz in einer Realweltanwendung" funded by the German Federal
Ministry for Economic Affairs and Climate Action.

The measurement data utilized in Scenario 2 has been taken from [here](https://github.com/fzi-forschungszentrum-informatik/tropical_precooling_environment).



## Citation

Please consider citing us if this environment and/or the accompanying [paper](https://dl.acm.org/doi/10.1145/3408308.3427614) was useful for your scientific work. You can use the following BibTex entry:

```latex
TODO: Add bibtex here after paper is online in ACM DL.
```

Other reference formats are provided [here](https://dl.acm.org/doi/10.1145/3408308.3427614) (export citation button).



## Contact

Please open an issue here on GitHub for any question or remark regarding the implementation. Please feel free to contact [David Wölfle](https://www.fzi.de/team/david-woelfle/) for all other inquiries.



## Copyright and license

Code is copyright to the FZI Research Center for Information Technology and released under the MIT [license](./LICENSE).

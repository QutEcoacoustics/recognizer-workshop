## Installation

### In conda

`conda create -y --name recognizer_workshop python=3.11 && conda activate recognizer_workshop`


We first need to install two dependencies 

Run the following command to install the perch-hoplite from the submodule. 

`pip install lib/perch-hoplite`

`pip install packages/baw-helpers`

Then

`pip install lib/recognizer-workshop`


## Run tests

First, `cd packages/recognizer-workshop`, then

`hatch env run -e test -- pytest -vv --capture=tee-sys`

This will: 
  - use hatch to run the tests in an isolated environment
  - print verbose as the test run AND capture the output so it will be dumped at the end for failed tests
    - this is useful if something is hanging due to staging server timeouts. 


# Running in jupyterhub

The recognizer workshop package doesn't need jupyter or jupyterhub to run, but it's built with deployment in a jupyterhub server in mind. 

To run in jupyterhub, some extra dependencies are needed:

- jupyter-server
- jupyter_bokeh
- ipywidgets (??) 





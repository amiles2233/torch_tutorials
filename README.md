# torch_tutorials

Torch is new to R, and it can be intimidating to use. The goal of this repo is to provide examples and skeleton scripts which will show how to to use torch in common ML scenarios (e.g. regression on tabular data, binary classification on text data, etc.). Moreso than vignettes, I hope to be able to identify and address common errors and roadblocks that users will come across.

Hopefully, data scientists will find this a useful resource for how to get up and running with torch.

I also feel obligated to note that model performance is not the focus of this repo, and as a result, the models perform poorly. The point of this repo is to take simple examples from reading in data, to training and evaluating models, and getting the output in a usable format.

## Tabular Input Data  

The tabular input data section of this repo all uses the Palmer Penguins data.

### Regression  

`tabular_regression.R`

### Binary Classification

`tabular_binary.R`

### Multi-class Classification

`tabular_multiclass.R`

## ..Coming Soon.. 

* Image Input Data (torchvision)

* Text Input Data (torchtextlib)

* Distributional Outputs (Torch distributions/Pyro)

* Multimodal Models (multiple inputs/outputs)

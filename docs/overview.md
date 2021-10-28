# Overview
`Sentience` is an AutoML framework for Deep Learning workflows. The plan is to design a custom script to define the workflow and let `Sentience` find the best hyperparameters for designing, training and evaluating a model. The hyperparameters to define a model is done by using `ModelSchema` object and then generate `Architecture` objects containing a model definition. The training and evaluation is currently done by hand. 

The structure of the project is as follows:-

* [core](core/overview.md) -> Package that contains the compiler framework. (WIP)
* [framework](framework/overview.md) -> This defines the basis of the entire sentience framework. (WIP)
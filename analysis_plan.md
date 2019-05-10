analysis plan
================
Jack Yan
5/10/2019

``` r
dat = read_csv("./data.csv")
```

    ## Parsed with column specification:
    ## cols(
    ##   .default = col_double(),
    ##   diagnosis = col_character(),
    ##   X33 = col_character()
    ## )

    ## See spec(...) for full column specifications.

Create a test dataset
=====================

Exploratory data analysis
=========================

### Clustering (Unsupervised learning)

### Correlation plots

### One-to-one relation between classes and covariates

Model building, assessing performance, and variable importance
==============================================================

Linear methods
--------------

### Logistic regression

### regularized logistic regression (glmnet)

### LDA

Non-linear methods
------------------

### QDA

### Naive Bayes

### KNN

Classification Trees and Ensemble methods
-----------------------------------------

### a single tree

### bagging

### random forests

### boosting (Adaboosting and binomial loss function)

Variable importance, Visualization, and interpretation (L7\_2.rmd): PDP, ICE, etc.
==================================================================================

Model comparison
================

Final model interpretation
==========================

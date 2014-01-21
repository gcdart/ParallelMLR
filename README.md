This README file describes how to use the parallel MLR package.

WARNING: The program was untested after I made some modifications ! If the
dataset or parameters dont read into memory, there is some modification to be
made to the Distributed Cache section of the code.
[emailing me is a better option]

## ACKNOWLEDGEMENTS


1. LBFGS.java and Msrch.java are the implementation
   of Limited Memory BFGS and associated line search by robert_dodier@yahoo.com	
   
# PACKAGE DESCRIPTION


1. This tool train a regularized Multiclass Logistic Regression for large number of classes.
2. This is especially useful, when the parameters of all classes cannot be held in memory.
3. THIS TOOL ASSUMES THE DATASET CAN BE FIT IN MEMORY.
    [ I plan to do extend this code to datasets which cannot be fit in memory
       in the near future, but no ideas as of yet. ]

## Dataset Format

Please use the Converter tool associated with the MulticlassClassifier package
to convert your dataset to the appropriate binary format.

Refer : gcdart.blogspot.com/2012/08/multiclass-classifer-with-hadoop.html

for more information.


## Training a Regularized Multiclass Logistic Regression Classifier

```
hadoop jar pmlr.jar org.pMLR.hadoop.TrainingDriver 
       -D gc.iterativemlr-train.startiter=0 
       -D gc.iterativemlr-train.lambda=.001 [reg-parameter-value]
       -D gc.iterativemlr-train.split=20 [a reasonable value for mapred.max.split.size to split the list class-labels file ]
       -D mapred.job.map.memory.mb=$mmem [make-sure dataset fits in memory]
       -D mapred.job.reduce.memory.mb=$mmem [make-sure dataset fits in memory]
       -D mapred.child.java.opts=$jvm [make-sure dataset fits in memory]
       -D mapred.task.timeout=0 
       -D mapred.map.max.attempts=100 
       -D gc.iterativemlr-train.iterations=100 [default outer-loop iterations]
       -D gc.iterativemlr-train.maxiter=200 [default inner-lbfgs MAX iterations]
       -D gc.iterativemlr-train.eps=.0001 [desired accuracy of lbfgs solution - IGNORED, refer lines 194 to 196 ]
       -D gc.dataset.train.loc=/datasets/dataset/seqfile/train/ [location of training dataset]
       -D gc.iterativemlr-train.input=/datasets/dataset/leaflabels/* [location of the file containing list of class-labels]
       -D gc.iterativemlr-train.output=/params/dataset/itermlr/weights/
       -D gc.iterativemlr-train-vparam.output=/params/dataset/itermlr/vparams/
       -D gc.iterativemlr-train-fvalues.dir=/params/dataset/itermlr/fvalues/
```

### Things to tweak

Here are some parameters to tweak for good performance and convergence,
(1) Regularization parameter : gc.iterativemlr-train.lambda
(2) Total number of iterations to run : iterativemlr-train.iterations
(3) The accuracy of inner lbfgs optimization. A heuristic has been implemented in lines 194 to 196.
    This is by no means a 'recommended' strategy. Please consider rewriting your own for your dataset.

## Testing a classifier


Please use the Testing tool associated with the MulticlassClassifier package
to convert your dataset to the appropriate binary format.

Refer : gcdart.blogspot.com/2012/08/multiclass-classifer-with-hadoop.html

for more information.

Siddharth Gopal (gcdart@gmail)
CMU, Pittsburgh

# Assam-Bhutan-Decision-Tree-Classifer
CSCI-720 

## Project Description

You are provided with a file of training data. This data has several attributes to select from.
They might include:
* The age of maturity the snowfolk (when their hair turns grey)
* Their height at maturity
* The length of their Bangs – the BangLn.
* The length of their tails, the TailLn
* Hair length ...
* If their earlobes are attached

Your goal is to classify the test data into *Assam ( -1 )* or *Bhutan ( +1 )*.<br>
Using the training data, your goal is to write one decision tree that classifies the results. Use only the attributes provided. No feature generation is allowed/required.


** Requirements

You write a program that implements the generic decision tree creation process, recursively.<br>
It creates a decision tree, using the decision tree algorithms we discussed in class. The output of this program is another program, a trained classifier.

The trained classifier program must be able to read in the *.csv file that is used to train the decision tree. So, the input to the decision tree training program is a flat file. For simplicity’s sake, you can write this program to make hard-coded assumptions. It can assume that there are only numeric attributes, and that there are only a certain number of attributes. So, don’t feel like you need to write a generic decision tree training program. The goal is to get the best accuracy on the validation data as possible. Again, the output of this trainer program is another program, Classifier.py

[![program_flow.png][program_flow.png]






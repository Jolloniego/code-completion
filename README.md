# MSc Dissertation Project: NLP for Code Completion
This repository contains the code used for running the experiments described in MSc dissertation associated with it. 

Instructions:
- Create a 'data' folder in the root of the project and a 'repos' folder under it.
- Place all cloned repositories in the 'repos' folder.
- Place the train, test and validation.txt files in the root of the 'data' folder.
  These files indicate how to split the data.

- Files in the repositories should be only .py files and normalised.
- To obtain the data, normalise and split it follow the instructions in: https://github.com/uclmr/pycodesuggest

- To build the vocabulary on the data specified by train.txt run build_vocabulary.py with appropriate arguments
  if the train.txt file was changed delete the vocab.p file and build the vocabulary again.

- To run an experiment just run 'python3 main_driver.py' with the desired arguments or write a script for it.
- Use python3 main_driver.py -h for detailed help on the possible arguments.

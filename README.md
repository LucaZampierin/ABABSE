# ABSE2
Code for the bachelor thesis 'An Unsupervised Approach for Aspect-Based Sentiment Analysis Using Attentional Neural Models' by Luca Zampierin (2021). The code is an adaptation of the scripts made available by Truşcǎ, Wassenberg, Frasincar & Dekker (2020). 

In this project, three novel unsupervised attentional neural network models for Aspect-Based Sentiment CLassification (ABSC) are introduced. The first two models, Aspect-Based Sentiment Extraction 1 (ABSE1) and Aspect-Based Sentiment Extraction 2 (ABSE2), are inspired by the work done by He, Lee, Ng, & Dahlmeier (2017). The third model, Unsupervised Left-Center-Right separated neural network with Rotatory attention (Uns-LCR-Rot), is an unsupervised adaptation of the LCR-Rot model proposed by Zheng and Xia (2018). ABSE2 is concluded to be the best performing one.

## Installation 

### Download the required files 
1. Download SemEval2015 task 12 Dataset: https://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools (for completeness we already included the datasets within the `externalData` folder)
2. Download SemEval2016 task 5 subtask 1 Dataset: https://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools (for completeness we already included the datasets within the `externalData` folder)
3. Download 300-dimensional GloVe word embeddings: https://nlp.stanford.edu/projects/glove/ (we used the embedding containing 1.9M vocabs and pre-trained on Common Crawl, but others can be experimented with too)

### Environment configuration
1. Download (if not yet available) a recent version of Python: https://www.python.org/downloads/. We used Python 3.8.6
2. Download a Python IDE, we suggest using PyCharm as this is the IDE we used inthis project and the following steps to set up the environemnt are speciic to PyCharm: https://www.jetbrains.com/pycharm/download/#section=windows.
3. Set-up a virtual environment in Pycharm:
    - Click the `Python interpreter selector` on the bottom right 
    - Select `Add Interpreter`
    - On the left-hand side of the window that appears, select `Virtual Environment`
    - Then, select the `New Environment` option where you will be asked to choose a file location and the base Python Interpreter (the one you downloaded at step 1.)
5. Download this software in a zip file and unzip it in the virtual environemnt directory. Conversely, you could also copy all the scripts in the virtual environment.
6. Activate the virtual environment py running the following line in the PyCharm Terminal: `C:\Users\...\venv\Scripts\activate` , where the dots indicate that you have to specify the directory where you saved your virual environment.
7. Install the libraries required to use this project by running the following line in the PyCharm Terminal: `pip install -r requirements.txt`
8. Run the command `python -m download en` in the Terminal in order to isntall the english space language package.

## Software explanation
- `main.py`: Script to run the training on the full training set and to test it on the original test set. The user can select the model that he/she wants to run by setting one of the booleans runABSE1, runABSE2, or runLCRROTU to True. Moreover, the user can modify the seed words used for the seed regularization of each model in this file. Run this file by running the line `main.py` in the Terminal.
- `main_hyper.py`: Script to run hyperparameter optimization (using the TPE algorithm) given a hyperparameter space that the user can modify. The user can select the model that he/she wants to tune by adjusting the `run_a_trial()` method. Run this file by running the line `main_hyper.py` in the Terminal.
- `main_cross.py`: Script to run the models but using cross-validation. The user can select the model that he/she wants to run by setting one of the booleans runABSE1, runABSE2, or runLCRROTU to True. Moreover, the user can modify the seed words used for the seed regularization of each model in this file. Run this file by running the line `main_cross.py` in the Terminal.
- `ABSE1.py`: TensorFlow implementation of the Aspect-Based Sentiment Extraction 1 (ABSE1) model.
- `ABSE2.py`: TensorFlow implementation of the Aspect-Based Sentiment Extraction 2 (ABSE2) model.
- `lcrModelU.py`: TensorFlow implementation of the Unsupervised Left-Center-Right separated neural network with Rotatory attention (Uns-LCR-Rot) model.
- `config.py`: contains different parameters that the user can modify. For example, the user can use this file to change the dataset used by changing the `year` value.
- `dataReader2016.py`: this file is used to read the original XML files into machine-readable files.
- `loadData.py`: this file loads the data to be used by the models.
- `att_layer.py`: this file defines the attention layers used in this project.
- `nn_layer.py`: this file defines the neural networks layers used in this project.
- `utils.py`: this file contains some more useful methods that are used over the project.

## Related Work
- He, R., Lee, W. S., Ng, H. T., & Dahlmeier, D. (2017). An unsupervised neural attention modelfor aspect extraction. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017)* (pp. 388–397). ACL.
- Truşcǎ, M. M., Wassenberg, D., Frasincar, F., & Dekker, R. (2020). A hybrid approach for aspect-based sentiment analysis using deep contextual word embeddings and hierarchical attention. In *20th International Conference on Web Engineering (ICWE 2020)* (Vol.12128, pp. 365–380). Springer.
- Zheng, S., & Xia, R. (2018). Left-center-right separated neural network for aspect-basedsentiment analysis with rotatory attention. *arXiv preprint arXiv:1802.00892*.







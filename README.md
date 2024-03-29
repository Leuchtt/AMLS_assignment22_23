# Introduction
This project focuses on the solution of four tasks: A1, A2, B1, and B2 based on machine learning algorithms. Logistic regression and CNN models are used for task A1; logistic regression and SVM models are used for task A2; decision tree and random forest models are used for tasks B1 and B2. The packages import, data preprocessing, model building, hyper-parameter tuning and cross-validation are implemented for each algorithm to realise, optimise and test the model. Each model for each task is imported in main.ipynb and the results of main.ipynb will show the test results of each model on separate test datasets.

# Role of each file
**A1:**<br />
**Logistic_regreesion.ipynb**---Builds logistic regression model to realise gender detection<br />
**CNN_train.ipynb**---Builds and train the CNN model for gender detection<br /> 
**CNN.ipynb**---Uses the model parameter saved to test the performance of the model on the test dataset<br />
**cnn_model.pt**---The file with CNN model parameters saved by CNN_train.ipynb 

**A2:**<br />
**Logistic_regreesion.ipynb**---Builds logistic regression model to realise smiling detection<br />
**SVM.ipynb**---Builds SVM model to realise smiling detection

**B1:**<br />
**decision_tree.ipynb**---Builds decision tree model to realise face shape recognition<br />
**random_forest.ipynb**---Builds random forest model to realise face shape recognition

**B2:**<br />
**sunglass_remove.ipynb**---Preprocesses data to classify sunglasses-wearing, and save the new train and test dataset without sunglasses<br />
**decision_tree.ipynb**---Builds decision tree model to realise face shape recognition<br />
**random_forest.ipynb**---Builds random forest model to realise face shape recognition
     
# Dependencies

  The code is programmed on python 3.9 based on **Jupyter Notebook**. The following packages are needed for this project:

    NumPy
    Pandas
    sklean
    cv2
    torch
    torchvision
    Matplotlib
    Seaborn
    PIL


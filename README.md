# Hybrid-Training-Data-Selection

Introduction 
Here, you find the source code for the approach proposed in the article " An Effective Software Cross-Project Fault Prediction Model for Quality Improvement” by Yogita Khatri and Sandeep Kumar Singh which is  currently under review at Science of Computer Programming. 

Requirements
•	Python
•	Anaconda (Run the code in Jupyter Notebook)
•	Windows 10
•	Install pyswarms package (Command to install: pip install pyswarms)

How to run it
•	Execute all steps in sequence

1.	Load the datasets ( code lines 16-82)
The first step is to load the datasets. As we are working on 62 datasets, therefore loading all 62 datasets in the first step. While loading the dataset we have set the paths of the datasets according to their location in our system. So, kindly change the path according to the location of the datasets in your system.

2.	Filter the datasets (code lines 92 -153)
As the first two features in each dataset refer to the project’s name and version and the third feature represents its instance’s name, therefore in the next step, we are filtering the datasets to remove these features as they are not relevant.

3.	Labeling of datasets (code lines 161-223)
This step consists of labeling all the instances in each dataset as faulty (by assigning an integer value ‘1’) if the bug count is greater than zero, otherwise as non-faulty by assigning an integer value ‘0’.

4.	Deletion of instances with zero LOC (code lines 231-294)
This step consists of deleting instances with zero LOC from all datasets. 

5.	Resetting the index in each dataset (code lines 300-362)
This step consists of resetting the indexes of all instances in each dataset after the removal of instances with zero LOC.

6.	Combining all datasets into a single dataset (code lines 370-371)
This step concatenates all datasets into a single dataset named ‘datasets’ with keys as their names. Kindly note that every project name consists of a number at the end. This number will help in differentiating the various versions of the same project. For example ‘ant1’ refers to the initial version (i.e. version 1.3) of the ‘ant’ project. Next, ‘ant2’ refers to the subsequent version (i.e version 1.4) of the ‘ant’ project, and so on.

7.	Making a list of all datasets (code line 378)
This step creates a list ‘list2’ of all datasets, where each of its elements is a particular dataset.

8.	Making a list of all dataset or project names (code line 386)
This step creates a list ‘list1’ of all dataset names, where each of its elements is the name of a  particular dataset.

9.	Defining a function ‘def euclidean_distance(row1, row2)’ (code lines 392-400)
This function calculates the Euclidean distance between two vectors ‘row1’ and  ‘row2’

10.	  Defining a function ‘def get_neighbors(train, test_row, num_neighbors)’ (code lines 404-413)
This function finds the ‘num_neighbors’ nearest neighbors of the target data instance ‘test_row’ from the ‘train’ dataset based on the Euclidean distance.

11.	Defining a function ‘KNN (target_data_label, target_data_index,list_related_releases)’ (code lines 425-466)
This function will generate the relevant source data for a particular target project, first by removing that target project and all its related versions from the combined dataset ‘datasets’ retrieved after step 6 and then calling the  ‘get_neighbors’ function to find the  10 nearest instances from ‘datasets’ for each target project instance. 

    Following are the three inputs to the function

    target_data_label: It specifies the name of the target project for which the source data is to be retrieved.

    target_data_index: It specifies the location of the target project in ‘list1’. For instance, the indexes of the target project ‘ant1’, ‘ant2’,      ‘ant3’,’ant4’, and ‘ant5’  in ‘list1’ are 0, 1, 2, 3, and 4 respectively.

    list_related_releases: It specifies the list of all related releases of the target project.

12.	 Defining a function ‘def compute_tp_tn_modified(actual,pred)’
(code lines 473-478)
This function accepts two inputs ‘actual’ and ‘pred’, which denote the actual and predicted labels and computes the confusion matrix. 

13.	 Defining a function ‘def smote(X_train,y_train)’ (code lines 485-490)
This function applies  SMOTE  to handle class imbalance issue. This function accepts two inputs defined as follows:

    X_train: training instances without labels on which SMOTE is to be applied

    y_train: labels of training instances

    The output returned is the balanced data with an equal number of faulty and non-faulty instances.

14.	Defining a function ‘def find_related_release(list1,target_data_index)’ (codelines 497-506)
This function finds the related versions of a target project with index ‘target_data_index’ from ‘list1’ (list1 already created in step 8).

15.	Defining a function ‘def sorted_df_epm1 (X_test, prob_score,  y_pred,y_test)’ (code lines 517-549)

    This function accepts four inputs defined as follows:

    X_test: It specifies the test data (target data) on which the actual performance of the classifier is measured.

    prob_score: It specifies the probability score obtained by the classifier on ‘X_test’

    y_pred: It specifies the prediction obtained by the classifier on ‘X_test’

    y_test: It specifies the actual labels of the instances contained in ‘X_test’

    This function returns a data frame containing all faulty and all non-faulty instances of  ‘X_test’ in order, sorted by {(prob_score/loc)*avg_cc } individually. This data frame will be used in three functions namely "PII_20, costeffort20, and IFA20"  to calculate effort-based performance measures (EPMs).

16.	Defining a function ‘def PII_20(target_data)’ (code lines 556-576)
This function accepts the sorted data frame returned from step 15 as input and computes the EPM ‘PII@20”.

17.	Defining a function ‘def costeffort20(target_data)’ (code lines 583-607)
This function accepts the sorted data frame returned from step 15 as input and computes the EPM ‘CostEffort@20”.

18.	Defining a function ‘def IFA20(target_data)’ (code lines 615-642)
This function accepts the sorted data frame returned from step 15 as input and computes the EPM ‘IFA”.

19.	Defining a function ‘def f_per_particle(m)’ (code lines 654-702)
This function accepts the binary feature vector  ‘m’ as input and calculates its fitness. It trains the Gaussian Naïve Bayes  classifier on the source data retrieved after the ‘smote’ function with features included in ‘m’ only, then validates it on 20% of the target data with features included in ‘m’ only, and then returns the performance in terms of G-measure.

20.	 Defining a function ‘def f(x)’ (code lines 714-730)
This function accepts the whole swarm ‘x’ as input and returns the fitness of all its particles.

21.	Create an empty list ‘avg_result’ to store the average performance over all datasets. (code line 737)

22.	 Now, the actual execution of the HTDS approach will start (code lines 751-850)

    For every target project, the instance selection phase is applied first which constitutes calling the 'KNN' function defined in step 11 and then the 'smote' function in sequence. After that, the feature selection phase will be executed 10 times for that target project, and the mean of the NEPMs and EPMs are calculated for that target project for which we have used the BPSO implementation of the ‘pyswarms’ library. So must include the ‘pyswarms’ package in the beginning. Kindly note that we have redefined the ‘f_per_particle(m)’ function as per our approach. The entire process is repeated for all 62 datasets and the average of the NEPMs and EPMs over all datasets are calculated. Then, the final output containing mean G-measure, MCC, PF, PII@20, CostEffort@20, and IFA is obtained in the respective order.



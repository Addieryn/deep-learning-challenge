Overview: 
The purpose of this analyzation is to help assess which applicants are most likely to be successful in their ventures by creating a model that can predict such a result.

Results: 

Data Preprocessing:
The Target variable for this model is the column "IS_SUCCESSFUL"
The Feature variables are "APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, ASK_AMT"           
The variables removed were "EIN, SPECIAL_CONSIDERATIONS"

Compiling, Training, and Evaluating the Model:
There ended up 3 hidden layers with 11 neurons, 1,400 neurons, and 15 neurons in that order. These numbers were chosen specifically to handle larger data, keeping the second and third relatively even, without going over 1mb in total.
The target model performance was achieved with these changes.
The major changes that lead to this result were: Adjusting which features were used in the learning, adding another layer, changing which activations that were used, and upping the total nodes, adjusting the epochs run and running it twice.

Summary: 
Overall this model was able to produce an accuracy of 79%-82% with a loss of around 50%.
I believe that with more cleaning the data prior to the model would help in its efficiency, along with the use of random forests that could help determine relevance in the data.
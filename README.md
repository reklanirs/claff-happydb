# claff-happydb

> [CL-Aff Shared Task] Happiness Ingredients Detection using Multi-Task Deep Learning

We propose a novel way of deploying deep multi-task learning models for the task of detecting disclosure and support. We calculate all possible logical relations among six labels, represented in a Venn diagram. Based on it, the six labels are distributed to multiple fragment clusters. Then, a multi-task deep neural network is built on the groups.



## Task Primary labels and an auxilary label

- Task A (label Agency) Binary label describing whether or not the author is in control.

- Task B (label Social) Binary label describing whether or not this happy moment involve people other than the author.

- Task C (label Concept) Auxiliary task. Concepts can have up to 15 possible values, and each happy moment can have multiple (up to 4) values.



## Preprocessing steps

- Split the sentences into word lists and omit all punctuation marks.

- Transform the sentences into sequences and pad them to become of the same length.

- Categorise Agency, Social, and Concepts.

Example:

| **Moment description**                   | **As I was walking to retrieve the  mail, I saw a neighbor walking their adorable dog.** |
| ---------------------------------------- | ------------------------------------------------------------ |
| **Agency**                               | Yes                                                          |
| **Social**                               | Yes                                                          |
| **Concept**                              | animals  \| exercise                                         |
| **After splitting:**                     | ['As',  'I', 'was', 'walking', 'to', 'retrieve', 'the', 'mail', 'I', 'saw', 'a',  'neighbor', 'walking', 'their', 'adorable', 'dog'] |
| **Assign each word a** **number:**       | {'I': 1, 'their': 2, 'walking': 3,  'mail': 4, 'saw': 5, 'was': 6, 'to': 7, 'a': 8, 'neighbor': 9, 'retrieve':  10, 'dog': 11, 'adorable': 12, 'the': 13, 'As': 14} |
| **Transform and padding** **ahead****:** | [0,  0, 0, 0, 14, 1, 6, 3, 7, 10, 13, 4, 1, 5, 8, 9, 3, 2, 12, 11] |
| **Categorise Concepts:**                 | (assume: animals, education,  exercise, food, party)  10100  |



## Deep Neural Network Model

**Structure from bottom to top** :

- Embedding Layer.  (Initialized with GloVe)
- 1D Convolutional Layer.
- Hard Parameter sharing Multi-task Learning Layers.



## **Experiments and Results**

##### Accuracies for Task A (Agency) and Task B (Social) on training data (split into 60% training, 20% validation, 20% test)

| **Model**                         | **Task  A** | **Task  B** |
| --------------------------------- | ----------- | ----------- |
| **CNN**                           | 71.2%       | 77.0%       |
| **CNN  + MTL**                    | 80.4%       | 85.8%       |
| **CNN  + MTL + GloVe (fixed)**    | 83.1%       | 87.6%       |
| **CNN + MTL + GloVe (trainable)** | 83.5%       | 89.2%       |



##### Precisions for Task A (Agency) and Task B (Social) on training data

| **Model**                         | **Task  A** | **Task  B** |
| --------------------------------- | ----------- | ----------- |
| **CNN**                           | 0.664       | 0.799       |
| **CNN  + MTL**                    | 0.759       | 0.860       |
| **CNN  + MTL + GloVe (fixed)**    | 0.815       | 0.876       |
| **CNN + MTL + GloVe (trainable)** | 0.813       | 0.892       |



##### Recalls for Task A (Agency) and Task B (Social) on training data

| **Model**                         | **Task  A** | **Task  B** |
| --------------------------------- | ----------- | ----------- |
| **CNN**                           | 0.689       | 0.778       |
| **CNN  + MTL**                    | 0.744       | 0.856       |
| **CNN  + MTL + GloVe (fixed)**    | 0.744       | 0.876       |
| **CNN + MTL + GloVe (trainable)** | 0.756       | 0.893       |



##### F1 scores for Task A (Agency) and Task B (Social) on training data

| **Model**                         | **Task  A** | **Task  B** |
| --------------------------------- | ----------- | ----------- |
| **CNN**                           | 0.670       | 0.767       |
| **CNN  + MTL**                    | 0.751       | 0.857       |
| **CNN  + MTL + GloVe (fixed)**    | 0.767       | 0.876       |
| **CNN + MTL + GloVe (trainable)** | 0.776       | 0.892       |



## Conclusion

Multi-task learning gets good results as it lowers the risk of overfitting on each task;

Task B achieves better results than Task A because the latter has unbalanced training data;

With trainable embeddings initialized with GloVe, the deep neural network can get better results.
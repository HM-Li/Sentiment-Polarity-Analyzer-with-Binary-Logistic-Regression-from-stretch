# sentiment-polarity-analyzer-with-binary-logistic-regression-from-stretch
A sentiment polarity analyzer with binary logistic regression build from stretch. Implemented a working Natural Language Processing (NLP) system, i.e., a sentiment polarity analyzer, using binary logistic regression. To determine whether a review is positive or negative using movie reviews as data. A basic feature engineering, through which you are able to improve the learner’s performance on this task. Wrote two programs: feature.py and lr.py to jointly complete the task.

# Materials
Materials Download the tar file from Autolab (“Download handout”). The tar file will contain all the data
that you will need in order to complete this assignment.  
The handout contains data from the Movie Review Polarity data set (for more details, see http://
www.cs.cornell.edu/people/pabo/movie-review-data/). Currently, the original data is
distributed as a collection of separate files (one movie review per file). In the Autolab handout, we have
converted this to a one line per example format consisting of the label 0 or 1 in the first column followed by
all the words in the movie review (with none of the line breaks) in the second column.
Each data point consists of a label (0 for negatives and 1 for positives) and a attribute (a set of words as a
whole). In the attribute, words are separated using white-space (punctuations are also separated with whitespace).
All characters are lowercased. No fancy pre-processing on the plain text is needed, because we have
already done most of the work for you in the handout. We also provide a dictionary file (dict.txt) to
limit the vocabulary to be considered in this assignments. Actually, this dictionary is constructed from the
training data. Examples of the dictionary content are as follows, where the second column is the index of
the word. Column one and column two are separated with white-space. Each line in dict.txt has the
format: word index\n.  
films 0  
adapted 1  
from 2  
comic 3  
Examples of the data are as follows.  
1 david spade has a snide , sarcastic sense of humor that works ...  
0 " mission to mars " is one of those annoying movies where , in ...  
1 anyone who saw alan rickman’s finely-realized performances in ...  
1 ingredients : man with amnesia who wakes up wanted for murder , ...  
1 ingredients : lost parrot trying to get home , friends synopsis : ...  
1 note : some may consider portions of the following text to be ...  
0 aspiring broadway composer robert ( aaron williams ) secretly ...  
0 america’s favorite homicidal plaything takes a wicked wife in " ...  
![program 1](https://github.com/HM-Li/ID3-Decision-Tree-from-scratch/blob/master/Inspect_Description.png)  

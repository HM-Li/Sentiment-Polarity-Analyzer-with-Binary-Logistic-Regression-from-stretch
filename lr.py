# %%

import numpy as np
import itertools
import os
import sys


class BasicPackage:

    def read_dicts_from_file(self, path, splitter):
        """read_dicts_from_file read files using path and name the splitter

        Args:
            path ([type]): [description]
            splitter ([type]): [description]

        Returns:
            [list of dicts]: ({value:index})
        """

        with open(path, 'r') as f:
            lines = f.read().splitlines()
            contents = [(line.split(splitter)) for line in lines]
        return {content[0]: content[1] for content in contents}

    def read_file(self, path, splitter):
        """read_file read a file from a path and return a list

        Args:
            path (string): os
            splitter (string): splitter
        """
        with open(path, 'r') as f:
            lines = f.read().splitlines()
            contents = np.array(list(line.split(splitter) for line in lines))
        return np.array(contents)

    def split_X_y(self, dataset):
        """split_X_y split X and y from the dataset and do transformation

        Args:
            dataset (list): formatted input data

        Returns:
            list: [description]
        """

        y = [int(line[0]) for line in dataset]
        # remove the additional '' caused by an extral '\t',convert elements to int
        X = [dict([map(int, element.split(':')) for element in line[1:-1]])
             for line in dataset]
        return X, y

    def write_file(self, list, path):
        """write_file #  write to a file

        Args:
            list ([type]): [description]
            path ([type]): [description]
        """

        with open(path, 'w') as file:
            for line in list:
                file.write(str(line)+'\n')


class SGDLogisticClassifier:
    coefficient = None
    interception = None
    iteration = None
    avg_neg_log_likelihoods_train = None
    avg_neg_log_likelihoods_vali = None
    # training_errors = None

    def __init__(self, learning_rate,  max_iteration, shaffle=False, vocabulary=None):
        # initialize the instance
        self.learning_rate = learning_rate
        self.shaffle = shaffle
        self.max_interation = max_iteration
        self.vocabulary = vocabulary

    def updateX(self, X):
        # add X0 at the rear of X
        X_0 = {(len(self.coefficient)-1): 1}
        [x.update(X_0) for x in X]

    def fit(self, X_train, X_vali):
        """fit build up the vocabulary; initiate coefficients; format X

        Args:
            X ([{index:value}]): a list of condensedly stored sparse feature space; the key should be integer

        returns: formatted X with X0 as 1
        """

        if self.vocabulary is None:
            # should be modified to talk raw inputs to form a vocabulary
            print("You haven't build your vocabulary.")
            return
        # fold the interception into the coefficient at the rear
        self.coefficient = np.zeros(len(self.vocabulary)+1)
        # note the correspondent index in coefficient equals to the key in x, the last one is x0
        X_train2 = X_train.copy()
        X_vali2 = X_vali.copy()
        self.updateX(X_train2)
        self.updateX(X_vali2)
        # in the future may consider shaffle both x and y sets if shaffle == true
        return X_train2, X_vali2

    def transform(self, X_train, y_train, X_vali, y_vali):
        # train model with X and y
        # initialize the iteration to be zero
        self.iteration = 0
        self.avg_neg_log_likelihoods_train = []
        self.avg_neg_log_likelihoods_vali = []
        # self.training_errors = []
        # update the initial log likelihood
        self.update_likelihood(X_train, y_train, X_vali, y_vali)
        for epoch in range(self.max_interation):
            # each epoch

            # update coefficients
            # sum_error = 0.0
            print("@Epoch "+str(epoch+1)+" | ", end='')
            for index, x in enumerate(X_train):
                print('-', end='')
                exp_dot_product = np.exp(
                    self.sparse_dot_product(x, self.coefficient))
                # calculate the behind part of the gradient descent
                partial_gradient_descent = - \
                    (y_train[index]-(exp_dot_product/(1.0+exp_dot_product)))
                # update each coefficient
                for index, xi in x.items():
                    self.coefficient[index] -= self.learning_rate * \
                        (xi*partial_gradient_descent)
            print()
            # update likelihood
            self.iteration += 1
            self.update_likelihood(X_train, y_train, X_vali, y_vali)

    def update_likelihood(self, X_train, y_train, X_vali, y_vali):
        # calculate the negtive log likelihood using the current coefficient
        avg_neg_log_likelihood_train = self.cal_avg_neg_log_likelihood(
            X_train, y_train, self.coefficient)
        avg_neg_log_likelihood_vali = self.cal_avg_neg_log_likelihood(
            X_vali, y_vali, self.coefficient)
        self.avg_neg_log_likelihoods_train.append(
            [self.iteration, avg_neg_log_likelihood_train])
        self.avg_neg_log_likelihoods_vali.append(
            [self.iteration, avg_neg_log_likelihood_vali])
        print('#Average negative log likelihood: train: ' +
              str(avg_neg_log_likelihood_train)+' | vali: '+str(avg_neg_log_likelihood_vali))

    def sparse_dot_product(self, dict1, vec2):
        """sparse_dot_product do sparse dot product for two vectors

        Args:
            dict1 (dict): a condensed saved sparse vector, start with the interception
            vec2 (array): a sparse vector
        """
        product = 0.0
        for k, v in dict1.items():
            product += vec2[k]*v
        return product

    def cal_avg_neg_log_likelihood(self, X, y, coef):
        # calcuate the average negative log likelihood
        neg_log_likelihood = 0.0
        for index, x in enumerate(X):
            dot_product = self.sparse_dot_product(x, coef)
            neg_log_likelihood += (-y[index]*dot_product +
                                   np.log(1+np.exp(dot_product)))
        return neg_log_likelihood/len(X)

    def predict_entry(self, x, coef):
        # take an entry and a list of coefficient predict the probability
        dot_product = self.sparse_dot_product(x, coef)
        return 1.0/(1.0+np.exp(-dot_product))

    def predict_prob(self, X):
        # add x0 at rear
        transformed_X = X.copy()
        self.updateX(transformed_X)
        # predict given X and trained coefficient and return a list of probabilities
        prediction = []
        [prediction.append(self.predict_entry(x, self.coefficient))
         for x in transformed_X]
        return prediction

    def predict(self, X):
        # use raw X
        # predict given X and trained coefficient and return a list of labels
        prediction_prob = self.predict_prob(X)
        return [1 if entry > 0.5 else 0 for entry in prediction_prob]

    # calculate the error rate for two lists and return the number
    def cal_error_rate(self, attribute_list, prediction_list):
        assert((type(attribute_list) is np.ndarray)
               & (type(prediction_list) is np.ndarray)
               & (attribute_list.size == prediction_list.size)), "Input list format problem."
        right_count = (attribute_list == prediction_list).sum()
        error_rate = (len(attribute_list)-right_count)/len(attribute_list)
        return error_rate


# %%
# load basic package
bp = BasicPackage()
# load arguments
formatted_train_input_path = os.path.join(
    'D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 4\handin', 'largeoutput', 'large_formatted_train_input.tsv'
)

formatted_vali_input_path = os.path.join(
    'D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 4\handin', 'largeoutput', 'large_formatted_valiation_input.tsv'
)
formatted_test_input_path = os.path.join(
    'D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 4\handin', 'largeoutput', 'large_formatted_test_input.tsv'
)
dictionary_path = os.path.join(
    'D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 4\handin', 'dict.txt')
train_out_path = os.path.join(
    'D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 4\handin', 'largeoutput', 'train_out.labels')
test_out_path = os.path.join(
    'D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 4\handin', 'largeoutput', 'test_out.labels')
metrics_out_path = os.path.join(
    'D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 4\handin', 'largeoutput', 'metrics_out.txt')
num_epoch = '60'
# formatted_train_input_path = os.path.join('./', sys.argv[1])

# formatted_vali_input_path = os.path.join('./', sys.argv[2])
# formatted_test_input_path = os.path.join('./', sys.argv[3])
# dictionary_path = os.path.join('./', sys.argv[4])
# train_out_path = os.path.join('./', sys.argv[5])
# test_out_path = os.path.join('./', sys.argv[6])
# metrics_out_path = os.path.join('./', sys.argv[7])
# num_epoch = sys.argv[8]

dictionary = bp.read_dicts_from_file(dictionary_path, ' ')
training_set = bp.read_file(formatted_train_input_path, '\t')
validation_set = bp.read_file(formatted_vali_input_path, '\t')
test_set = bp.read_file(formatted_test_input_path, '\t')

X_train, y_train = bp.split_X_y(training_set)

X_valid, y_valid = bp.split_X_y(validation_set)

X_test, y_test = bp.split_X_y(test_set)

# %%
# initialize a classifier
clf = SGDLogisticClassifier(0.1, int(num_epoch), False, dictionary)
X_train2, X_valid2 = clf.fit(X_train, X_valid)
clf.transform(X_train2, y_train, X_valid2, y_valid)
print(clf.avg_neg_log_likelihoods_train)
print(clf.avg_neg_log_likelihoods_vali)

# %%
prediction_test = clf.predict(X_test)
prediction_train = clf.predict(X_train)
# calculate metrics
train_error = clf.cal_error_rate(np.array(y_train), np.array(prediction_train))
test_error = clf.cal_error_rate(np.array(y_test), np.array(prediction_test))
error_list = ['error(train): '+str(train_error),
              'error(test): ' + str(test_error)]
# write metrics
bp.write_file(error_list, metrics_out_path)
bp.write_file(prediction_train, train_out_path)
bp.write_file(prediction_test, test_out_path)


# # %% create chart
# import seaborn as sns
# import pandas as pd
# log_1 = np.array(clf.avg_neg_log_likelihoods_train[1:])
# log_2 = np.array(clf.avg_neg_log_likelihoods_vali[1:])
# log = pd.DataFrame(log_1, columns=['epoch', 'avg_neg_log_likelihood'])
# log2 = pd.DataFrame(log_2, columns=['epoch', 'avg_neg_log_likelihood'])
# log['label'] = "train"
# log2['label'] = "validation"
# log3 = pd.concat([log, log2])
# plot = sns.lmplot(x="epoch", y="avg_neg_log_likelihood",
#                   hue="label", data=log3, fit_reg=False, size=8)
# plot.savefig('./handin/large_200_1.png')
# # %%
# import pickle
# with open(os.path.join('./handin', 'log_large_200_1.pkl'), 'wb') as f:
#     pickle.dump(log3, f)

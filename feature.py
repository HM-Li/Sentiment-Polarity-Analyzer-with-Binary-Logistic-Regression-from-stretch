# %%
import numpy as np
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
            contents = [line.split(splitter) for line in lines]
        return np.array(contents)

    def save_X_y(self, X, y, path):
        # pair X and y back together and print out to a tsv file
        with open(path, 'w') as file:
            index = 0
            for label in y:
                file.write(label+'\t')
                # in this assignment, we only consider whether a word exists, so we just put a 1 in behind
                [file.write(str(x[0])+':'+str(1)+'\t') for x in X[index]]
                file.write('\n')
                index += 1


class CountVectorizer:
    def __init__(self, feature_flag, max_tf=None, vocabulary=None):
        """__init__ [summary]

        Args:
            feature_flag ([type]): define the max term frequency threshold for model 2
            max_tf ([type], optional): Defaults to None. 1 for model 1; 2 for model 2
            vocabulary (dict, optional): Defaults to None. input the vocabulary {value:index}
        """

        self.max_tf = max_tf
        self.feature_flag = feature_flag
        self.vocabulary = vocabulary

    def fit(self, raw_inputs=None):
        """fit prepare the vocabulary using the raw data

        Args:
            raw_inputs (list): list of texts
        """

        # for this assignment, the vocabulary is inputted by the user at defult
        # in the future this function should be updated to take the raw input and build a vocabulary
        if self.vocabulary is None:
            print('No vocabulary has been inputted.')

    def transform(self, X):
        """transform transform the raw inputs and return a condensed document term:tf matrix

        Args:
            X (List): a list of string
        """
        outputs = []
        raw_inputs = X.copy()
        splitted = self.prepare_raw(raw_inputs)
        # voc = np.fromiter(self.vocabulary.keys(), dtype='<U5')
        # change a way to extract key because the upper one could be restricted by the length of string
        voc = [key for key in self.vocabulary.keys()]
        if self.feature_flag == '1':
            for entry in splitted:
                word_value_pair = self.get_invoc_unique_words(entry, voc)
                index_value_pair = self.match_index(word_value_pair)
                outputs.append(index_value_pair)
        else:
            for entry in splitted:
                word_value_pair = self.get_invoc_words(entry, voc)
                trimmed_pair = self.trim_words(word_value_pair)
                index_value_pair = self.match_index(trimmed_pair)
                outputs.append(index_value_pair)
        return np.array(outputs)

    def get_invoc_unique_words(self, target_list, voc):
        # get unique words from the target list that belongs to the vocabulary
        # return word:
        inter = np.intersect1d(target_list, voc)
        # map
        pair = list(zip(inter, [1]*inter.size))
        return pair

    def get_invoc_words(self, target_list, voc):
        # return words from the target list that belongs to the voc
        inter = np.array(target_list)[np.in1d(np.array(target_list), voc)]
        # map reduce
        unique, counts = np.unique(inter, return_counts=True)
        pair = list(zip(unique, counts))
        return pair

    def trim_words(self, target_list):
        # trim words using the default max_tf
        trimed_list = list(filter(lambda x: x[1] < self.max_tf, target_list))
        return trimed_list

    def match_index(self, target_list):
        # replace words with index
        indexed_list = [(self.vocabulary.get(pair[0]), pair[1])
                        for pair in target_list]
        return indexed_list

    def prepare_raw(self, raw_input):
        # split each row to a list
        splitted = [row.split(' ') for row in raw_input]
        return splitted


# %%
# %time
# load basic package
bp = BasicPackage()
# load arguments
train_input_path = os.path.join(
    'D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 4\handin', 'largedata', 'largetrain_data.tsv')
valiation_input_path = os.path.join(
    'D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 4\handin', 'largedata', 'largevalid_data.tsv')
test_input_path = os.path.join(
    'D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 4\handin', 'largedata', 'largetest_data.tsv')
dictionary_path = os.path.join(
    'D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 4\handin', 'dict.txt')
formatted_train_out_path = os.path.join(
    'D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 4\handin', 'largeoutput', 'large_formatted_train_input.tsv'
)
formatted_vali_out_path = os.path.join(
    'D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 4\handin', 'largeoutput', 'large_formatted_valiation_input.tsv'
)
formatted_test_out_path = os.path.join(
    'D:\MicFile\百度云同步盘\Document\Lectures\Semester 3\Machine Learning\Homeworks\Homework 4\handin', 'largeoutput', 'large_formatted_test_input.tsv'
)
feature_flag = '1'
# train_input_path = os.path.join('./', sys.argv[1])
# valiation_input_path = os.path.join('./', sys.argv[2])
# test_input_path = os.path.join('./', sys.argv[3])
# dictionary_path = os.path.join('./', sys.argv[4])
# formatted_train_out_path = os.path.join('./', sys.argv[5])
# formatted_vali_out_path = os.path.join('./', sys.argv[6])
# formatted_test_out_path = os.path.join('./', sys.argv[7])
# feature_flag = sys.argv[8]

# load files
dictionary = bp.read_dicts_from_file(dictionary_path, ' ')
training_set = bp.read_file(train_input_path, '\t')
valiation_set = bp.read_file(valiation_input_path, '\t')
test_set = bp.read_file(test_input_path, '\t')
# split X and y
y_train = training_set.T[0]
X_train = training_set.T[1].T
y_vali = valiation_set.T[0]
X_vali = valiation_set.T[1].T
y_test = test_set.T[0]
X_test = test_set.T[1].T
cv = CountVectorizer(feature_flag, 4, dictionary)
cv.fit()
output_train = cv.transform(X_train)
output_vali = cv.transform(X_vali)
output_test = cv.transform(X_test)
bp.save_X_y(output_train, y_train, formatted_train_out_path)
bp.save_X_y(output_vali, y_vali, formatted_vali_out_path)
bp.save_X_y(output_test, y_test, formatted_test_out_path)

"""
    Assignment 2 - Testing
    FIT2004: Algorithms and Data Structures
    Ahmad Abu-Shaqra
    20/10/2023

"""

import unittest
import time
from assignment2 import Trie, load_dictionary, allocate

dictionary = load_dictionary("test_dictionary.txt")
myTrie = Trie(dictionary)

# 1: Customized Auto-Complete

class TestingQ1(unittest.TestCase):

    def test_01(self):

        # initialising test
        prefix = ''
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'acpunj')
        self.assertEqual(definition, 'fpowsspkobdt')
        self.assertEqual(frequency, 1000000)

    def test_02(self):

        # initialising test
        prefix = 'skzv'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'skzv')
        self.assertEqual(definition, 'kmcfppjgzcmoxrxy')
        self.assertEqual(frequency, 2)

    def test_03(self):

        # initialising test
        prefix = 'evtfq'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, None)
        self.assertEqual(definition, None)
        self.assertEqual(frequency, 0)

    def test_04(self):

        # initialising test
        prefix = 'siczn'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, None)
        self.assertEqual(definition, None)
        self.assertEqual(frequency, 0)

    def test_05(self):

        # initialising test
        prefix = 'mvn'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'mvnyglg')
        self.assertEqual(definition, 'fuavqfalafdeyp')
        self.assertEqual(frequency, 66)

    def test_06(self):

        # initialising test
        prefix = 'pyolv'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, None)
        self.assertEqual(definition, None)
        self.assertEqual(frequency, 0)

    def test_07(self):

        # initialising test
        prefix = 'u'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'uknxzlja')
        self.assertEqual(definition, 'fykikmlfnflnnpqe')
        self.assertEqual(frequency, 38574)

    def test_08(self):

        # initialising test
        prefix = 'x'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'xfbp')
        self.assertEqual(definition, 'jfuysrybzebffexh')
        self.assertEqual(frequency, 38656)

    def test_09(self):

        # initialising test
        prefix = 'oh'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'ohewpn')
        self.assertEqual(definition, 'vwnkqhyjhwdfcgmuhldvlzjj')
        self.assertEqual(frequency, 1484)

    def test_10(self):

        # initialising test
        prefix = 'elwm'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'elwmeh')
        self.assertEqual(definition, 'pdbvzdeyyhapxvbsqgbixast')
        self.assertEqual(frequency, 2)

    def test_11(self):

        # initialising test
        prefix = 'rgn'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'rgnc')
        self.assertEqual(definition, 'pdekrtvfvuctckfh')
        self.assertEqual(frequency, 52)

    def test_12(self):

        # initialising test
        prefix = 'wgdfs'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, None)
        self.assertEqual(definition, None)
        self.assertEqual(frequency, 0)

    def test_13(self):

        # initialising test
        prefix = 'jmerc'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, None)
        self.assertEqual(definition, None)
        self.assertEqual(frequency, 0)

    def test_14(self):

        # initialising test
        prefix = 'e'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'ealmuzc')
        self.assertEqual(definition, 'mwtxdhvjmvpoud')
        self.assertEqual(frequency, 38672)

    def test_15(self):

        # initialising test
        prefix = 'tayed'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, None)
        self.assertEqual(definition, None)
        self.assertEqual(frequency, 0)

    def test_16(self):

        # initialising test
        prefix = 'iv'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'ivsmrfjj')
        self.assertEqual(definition, 'hhnfuskbjjoxoljrqeubfnbxljvbdqfk')
        self.assertEqual(frequency, 1460)

    def test_17(self):

        # initialising test
        prefix = 'sfn'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'sfnss')
        self.assertEqual(definition, 'epdgufbswwtzbpz')
        self.assertEqual(frequency, 65)

    def test_18(self):

        # initialising test
        prefix = 'tuye'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'tuyejkk')
        self.assertEqual(definition, 'hkagrwbvedswjascfkmqr')
        self.assertEqual(frequency, 3)

    def test_19(self):

        # initialising test
        prefix = 'qby'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'qbywh')
        self.assertEqual(definition, 'ewbexpvvwg')
        self.assertEqual(frequency, 66)

    def test_20(self):

        # initialising test
        prefix = 'f'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'fakescgq')
        self.assertEqual(definition, 'dbsvngmcbkxhualaisewbzdv')
        self.assertEqual(frequency, 38595)

    def test_21(self):

        # initialising test
        prefix = 'i'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'ibpqgpwu')
        self.assertEqual(definition, 'aebkglnpsnywzdmilqpsaogy')
        self.assertEqual(frequency, 38478)

    def test_22(self):

        # initialising test
        prefix = 'jj'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'jjlkd')
        self.assertEqual(definition, 'ckyilzcccz')
        self.assertEqual(frequency, 1541)

    def test_23(self):

        # initialising test
        prefix = 'asp'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'aspy')
        self.assertEqual(definition, 'usjdbkbx')
        self.assertEqual(frequency, 53)

    def test_24(self):

        # initialising test
        prefix = 'osrq'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'osrqqy')
        self.assertEqual(definition, 'mwxatpeagyoe')
        self.assertEqual(frequency, 1)

    def test_25(self):

        # initialising test
        prefix = 'dh'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'dhgor')
        self.assertEqual(definition, 'xpkzsljayzeacocfbglg')
        self.assertEqual(frequency, 1477)

    def test_26(self):

        # initialising test
        prefix = 'w'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'wcwnxu')
        self.assertEqual(definition, 'mgajueygfromzfpwfqnesdgv')
        self.assertEqual(frequency, 38373)

    def test_27(self):

        # initialising test
        prefix = 'nqxgq'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, None)
        self.assertEqual(definition, None)
        self.assertEqual(frequency, 0)

    def test_28(self):

        # initialising test
        prefix = 'ukism'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, None)
        self.assertEqual(definition, None)
        self.assertEqual(frequency, 0)

    def test_29(self):

        # initialising test
        prefix = 'gp'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'gpudjfva')
        self.assertEqual(definition, 'zonqyqrizwvltxoblcxudnfacrpwxulq')
        self.assertEqual(frequency, 1553)

    def test_30(self):

        # initialising test
        prefix = 'whm'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'whmkejl')
        self.assertEqual(definition, 'ahhvzmqfwhkopieqrpeko')
        self.assertEqual(frequency, 64)

    def test_31(self):

        # initialising test
        prefix = 'sbm'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'sbmqtiyz')
        self.assertEqual(definition, 'zbgalfixhaovaqxg')
        self.assertEqual(frequency, 52)

    def test_32(self):

        # initialising test
        prefix = 'b'
        word, definition, frequency = myTrie.prefix_search(prefix)

        # testing
        self.assertEqual(word, 'bhon')
        self.assertEqual(definition, 'oizztdgiohcn')
        self.assertEqual(frequency, 38616)


# 2: A Weekend Getaway

class TestingQ2(unittest.TestCase):

    def test_01(self):

        # initialising test
        preferences = [[0], [1], [0,1], [0, 1], [1, 0], [1], [1, 0], [0, 1], [1]]
        licences = [1, 4, 0, 5, 8]
        cars = allocate(preferences, licences)

        # testing
        expected = [[[0, 4, 2, 3 ,6], [1, 5, 8, 7]],
                    [[0, 4, 2, 3, 7], [1, 5, 8, 6]],
                    [[0, 4, 2, 6, 7], [1, 5, 8, 3]],
                    [[0, 4, 3, 6, 7], [1, 5, 8, 2]],
                    [[0, 4, 2, 3], [1, 5, 8, 6, 7]],
                    [[0, 4, 2, 6], [1, 5, 8, 3, 7]],
                    [[0, 4, 2, 7], [1, 5, 8, 3, 6]],
                    [[0, 4, 3, 6], [1, 5, 8, 2, 7]],
                    [[0, 4, 3, 7], [1, 5, 8, 2, 6]],
                    [[0, 4, 6, 7], [1, 5, 8, 2, 3]]]
        self.assertEqual(len(cars), len(expected[0]))
        output = find_chosen_output(cars, expected)
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[output][i])

    def test_02(self):

        # initialising test
        preferences = [[], [], [], [], [], [], [], []]
        licences = [4, 2, 0]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_03(self):

        # initialising test
        preferences = [[1, 0], [0, 1], [1, 0], [], [1], [0], [1]]
        licences = [6, 5, 2]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_04(self):

        # initialising test
        preferences = [[0], [0], [0], [0], [0]]
        licences = [0, 4, 3, 2, 1]
        cars = allocate(preferences, licences)

        # testing
        expected = [[0, 1, 2, 3, 4]]
        self.assertEqual(len(cars), len(expected))
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[i])

    def test_05(self):

        # initialising test
        preferences = [[0], [0], [0], [0], [0]]
        licences = [0]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_06(self):

        # initialising test
        preferences = [[0], [0], [0], [0], [0]]
        licences = []
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_07(self):

        # initialising test
        preferences = [[0], [0], [0], [0], [0], [1]]
        licences = [5, 4, 3, 2, 1, 0]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_08(self):

        # initialising test
        preferences = [[0]]
        licences = [0]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)
    
    def test_09(self):

        # initialising test
        preferences = [[0], [0], [0], [0], [0], [0]]
        licences = [1, 4, 3]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_10(self):

        # initialising test
        preferences = [[0], [0, 1], [0], [1, 0], [1], [1]]
        licences = [4, 2, 0, 5]
        cars = allocate(preferences, licences)

        # testing
        expected = [[[0, 1, 2, 3], [4, 5]],
                    [[0, 1, 2], [3, 4, 5]],
                    [[0, 2, 3], [1, 4, 5]],
                    [[0, 2], [1, 3, 4, 5]]]
        self.assertEqual(len(cars), len(expected[0]))
        output = find_chosen_output(cars, expected)
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[output][i])
        
    def test_11(self):

        # initialising test
        preferences = [[0, 1], [0, 1], [1, 0], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]
        licences = [10, 1, 7, 3, 5, 0]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_12(self):

        # initialising test
        preferences = [[0, 1, 2], [0, 2], [0], [0, 1], [2, 1], [0, 1], [0], [2], [1, 2], [0], [0, 2], [1]]
        licences = [4, 2, 0, 1]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)
    
    def test_13(self):

        # initialising test
        preferences = [[1], [0], [0, 1], [1, 0], [0, 1], [1, 0], [1], [1], [0], [0, 1]]
        licences = [6, 3, 4, 9, 1]
        cars = allocate(preferences, licences)

        # testing
        expected = [[[1, 8, 4, 5, 9], [0, 6, 7, 2, 3]],
                    [[1, 8, 3, 5, 9], [0, 6, 7, 2, 4]],
                    [[1, 8, 3, 4, 5], [0, 6, 7, 2, 9]],
                    [[1, 8, 2, 5, 9], [0, 6, 7, 3, 4]],
                    [[1, 8, 2, 4, 9], [0, 6, 7, 3, 5]],
                    [[1, 8, 2, 4, 5], [0, 6, 7, 3, 9]],
                    [[1, 8, 2, 3, 9], [0, 6, 7, 4, 5]],
                    [[1, 8, 2, 3, 5], [0, 6, 7, 4, 9]],
                    [[1, 8, 2, 3, 4], [0, 6, 7, 5, 9]]]
        self.assertEqual(len(cars), len(expected[0]))
        output = find_chosen_output(cars, expected)
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[output][i])

    def test_14(self):

        # initialising test
        preferences = [[0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [2], [2]]
        licences =  [0, 1, 5, 6, 10, 11]
        cars = allocate(preferences, licences)

        # testing
        expected = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11]]
        self.assertEqual(len(cars), len(expected))
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[i])

    def test_15(self):

        # initialising test
        preferences = [[0], [1], [0,1], [0], [1, 0], [1], [1], [0, 1], [1]]
        licences = [1, 4, 0, 5, 8]
        cars = allocate(preferences, licences)

        # testing
        expected = [[[0, 4, 2, 3], [1, 5, 8, 6, 7]],
                    [[0, 4, 3, 7], [1, 5, 8, 6, 2]],
                    [[0, 4, 2, 3, 7], [1, 5, 8, 6]],
                    [[0, 4, 3], [1, 5, 8, 6, 2, 7]]]
        self.assertEqual(len(cars), len(expected[0]))
        output = find_chosen_output(cars, expected)
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[output][i])

    def test_16(self):

        # initialising test
        preferences = [[1], [0, 1], [0], [1, 0], [0], [0]]
        licences = [0, 1, 2, 4]
        cars = allocate(preferences, licences)

        # testing
        expected = [[[2, 3, 4, 5], [0, 1]],
                    [[2, 4, 5], [0, 1, 3]]]
        self.assertEqual(len(cars), len(expected[0]))
        output = find_chosen_output(cars, expected)
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[output][i])
    
    def test_17(self):

        # initialising test
        preferences = [[0], [0]]
        licences = [1, 0]
        cars = allocate(preferences, licences)

        # testing
        expected = [[0, 1]]
        self.assertEqual(len(cars), len(expected))
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[i])

    def test_18(self):

        # initialising test
        preferences = [[0], [0], []]
        licences = [2]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_19(self):

        # initialising test
        preferences = [[0], [0], [0], [0], [0]]
        licences = [2, 1, 4]
        cars = allocate(preferences, licences)

        # testing
        expected = [[0, 1, 2, 3, 4]]
        self.assertEqual(len(cars), len(expected))
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[i])

    def test_20(self):

        # initialising test
        preferences = [[0], [], [0], [0]]
        licences = [0, 2, 1]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)
    
    def test_21(self):

        # initialising test
        preferences = [[0], [1], [1, 0], [1], [0], [1], [0]]
        licences = [3, 4, 1, 2]
        cars = allocate(preferences, licences)

        # testing
        expected = [[0, 4, 2, 6], [1, 3, 5]]
        self.assertEqual(len(cars), len(expected))
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[i])

    def test_22(self):

        # initialising test
        preferences = [[]]
        licences = []
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_23(self):

        # initialising test
        preferences = [[0], [0], [0], [0]]
        licences = [0, 2]
        cars = allocate(preferences, licences)

        # testing
        expected = [[0, 1, 2, 3]]
        self.assertEqual(len(cars), len(expected))
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[i])

    def test_24(self):

        # initialising test
        preferences = [[0], [0], [0]]
        licences = [2, 0, 1]
        cars = allocate(preferences, licences)

        # testing
        expected = [[0, 1, 2]]
        self.assertEqual(len(cars), len(expected))
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[i])
    
    def test_25(self):

        # initialising test
        preferences = [[0], [0], [0], [0], [0, 1], [1]]
        licences = [0, 1, 4, 5]
        cars = allocate(preferences, licences)

        # testing
        expected = [[0, 1, 2, 3], [4, 5]]
        self.assertEqual(len(cars), len(expected))
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[i])

    def test_26(self):

        # initialising test
        preferences = [[0], [0], [0], [0], [0, 1], [1]]
        licences = [0, 1, 5]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_27(self):

        # initialising test
        preferences = [[0, 2], [1], [1, 0], [0], [0], [1], [0], [2], [2], [2], [2, 1], [1]]
        licences = [0, 3, 2, 5, 10, 9]
        cars = allocate(preferences, licences)

        # testing
        expected = [[[0, 3, 4, 6], [2, 5, 1, 11], [10, 9, 8, 7]], 
	                [[2, 3, 4, 6], [10, 5, 1, 11], [0, 9, 8, 7]]]
        self.assertEqual(len(cars), len(expected[0]))
        output = find_chosen_output(cars, expected)
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[output][i])

    def test_28(self):

        # initialising test
        preferences = [[3, 0, 1], [3, 1, 2, 0], [0, 3, 2, 1], [2, 1, 3, 0], [3, 2, 1, 0], [3, 1, 0, 2], [3, 2, 0, 1], [2, 3, 0], [0, 3, 1, 2], [2, 0, 3, 1], [1, 2, 0, 3], [2, 3, 1, 0], [1, 2, 3, 0], [3, 2, 0], [3, 2], [3, 2, 0], [0, 2, 1, 3], [2, 0, 3, 1]]
        licences = [0, 5, 10, 15, 7, 11, 2]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)
    
    def test_29(self):

        # initialising test
        preferences = [[0], [1], [0], [0, 1], [1], [0], [1, 0]]
        licences = [1, 3, 2, 6]
        cars = allocate(preferences, licences)

        # testing
        expected = [[[0, 2, 5, 3], [1, 4, 6]],
	                [[0, 2, 5, 6], [1, 4, 3]]]
        self.assertEqual(len(cars), len(expected[0]))
        output = find_chosen_output(cars, expected)
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[output][i])

    def test_30(self):

        # initialising test
        preferences = [[0, 1], [1, 0], [0, 1], [0, 1], [0], [1, 0]]
        licences = [5, 3]
        cars = allocate(preferences, licences)

        # testing
        self.assertEqual(cars, None)

    def test_31(self):

        # initialising test
        preferences = [[1, 0], [0, 1], [0], [0], [0], [1], [1], [0, 1], [1], [1, 2], [2], [2]]
        licences = [0, 1, 5, 6, 9, 10]
        cars = allocate(preferences, licences)

        # testing
        expected = [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11]]
        self.assertEqual(len(cars), len(expected))
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[i])

    def test_32(self):

        # initialising test
        preferences = [[1, 0, 2], [0], [0], [0], [0], [2, 1, 0], [1], [1], [1], [0, 1, 2], [2], [2]]
        licences = [0, 1, 5, 6, 9, 10]
        cars = allocate(preferences, licences)

        # testing
        expected = [[[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11]],
                    [[5, 1, 2, 3, 4], [9, 6, 7, 8], [0, 10, 11]],
                    [[9, 1, 2, 3, 4], [0, 6, 7, 8], [5, 10, 11]],
                    [[0, 1, 2, 3, 4], [9, 6, 7, 8], [5, 10, 11]],
                    [[5, 1, 2, 3, 4], [0, 6, 7, 8], [9, 10, 11]],
                    [[9, 1, 2, 3, 4], [5, 6, 7, 8], [0, 10, 11]]]
        self.assertEqual(len(cars), len(expected[0]))
        output = find_chosen_output(cars, expected)
        for i in range(len(cars)):
            self.assertCountEqual(cars[i], expected[output][i])


# Helper Functions

def load_dictionary(filename):
    infile = open(filename, encoding = 'utf-8')
    word, frequency = "", 0
    aList = []
    for line in infile:
        line.strip()
        if line[0:4] == "word":
            line = line.replace("word: ","")
            line = line.strip()
            word = line            
        elif line[0:4] == "freq":
            line = line.replace("frequency: ","")
            frequency = int(line)
        elif line[0:4] == "defi":
            index = len(aList)
            line = line.replace("definition: ","")
            definition = line.replace("\n","")
            aList.append([word,definition,frequency])
    return aList

def find_chosen_output(cars, expected):
    for i in range(len(expected)):
        is_output = True
        for j in range(len(expected[i])):
            if set(cars[j]) != set(expected[i][j]):
                is_output = False
        if is_output:
            return i
    return 0


# Run Tests

if __name__ == '__main__':
    timer = time.time()
    Dictionary = load_dictionary("Dictionary.txt")
    print(f"\nDictionary Load Time: {round(time.time() - timer, 3)}s")
    timer = time.time()
    myTrie = Trie(Dictionary)
    print(f"Trie Construction Time: {round(time.time() - timer, 3)}s\n")
    time.sleep(2)
    unittest.main()
    
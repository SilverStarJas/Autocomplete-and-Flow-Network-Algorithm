import math
from collections import deque

### DO NOT CHANGE THIS FUNCTION
def load_dictionary(filename):
    infile = open(filename)
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


# 1. Customized Auto-Complete
class TrieNode:
    def __init__(self, key, data):
        """
        Function description: Initialises a trie node with all the necessary attributes

        Input:
            - key: the character that represents the node
            - data: a list that stores information about the word represented in the node

        Attributes: 
            - key: the character that represents the node
            - data: a list that stores information about the word represented in the node
            - next: a list to store references to child nodes
            - highest_freq: the highest frequency of a word reachable from this node
            - highest_data: information (definition and frequency) about the most frequent word
            - num_matches: the number of matches for words with the prefix represented by this node

        Time complexity: O(1)
        Aux space complexity: O(1) 
        """
        self.key = key
        self.data = data
        self.next = [None]*26  # no whitespaces so use None instead
        self.highest_freq = 0  # highest frequency of the child node/word that can be reached from current
        self.highest_data = []  # stores the info about the highest frequency word
        self.num_matches = 0


    def get_index(self, letter):
        """
        Function description: Retrieves the index of a letter and maps it to an integer from 0-25 (corresponding to the 
        lower case letters in the English alphabet).

        Input:
            - letter (str): a lowercase English letter

        Output: Returns an integer that represents the index of the letter.

        Time complexity: O(1)
        Aux space complexity: O(1) 
        """
        if letter.isalpha() and letter.islower():
            return (ord(letter) - ord('a')) # maps letter to an integer from 0-25


    def get_child_node(self, letter):
        """
        Function description: Retrieves the child node of a given letter, or creates one if a child node doesn't exist.

        Input:
            - letter (str): a lowercase English letter

        Output: Returns a TrieNode which is the child node of the given letter.

        Time complexity: O(1)
        Aux space complexity: O(1) 
        """
        index = self.get_index(letter)
        # check if child node exists - if not, then create one with None 
        if not self.next[index]:
            self.next[index] = TrieNode(letter, None)

        return self.next[index]  # return child node that represents the letter


    def node_search(self, letter):
        """
        Function description: Searches the trie for a node that represents the given letter.

        Input:
            - letter (str): a lowercase English letter.

        Output: Returns a TrieNode that represents the given letter, or None if it isn't found.

        Time complexity: O(1)
        Aux space complexity: O(1) 
        """
        index = self.get_index(letter)
        return self.next[index] 


class Trie:
    def __init__(self, Dictionary) -> None:
        """
        Function description: Initialises a prefix trie containing the words in the dictionary and their information 
        (definition and frequency). There are conditions to check if the current word has a higher frequency than the most
        frequent word so far and updates the node. If frequencies are the same, then it checks for alphabetical order and
        should update to the word that is first alphabetically. As it iterates through the letters in a word, num_matches is
        incremented to show how many words share the same prefix.

        Input:
            - Dictionary (list): contains tuples [word, definition, frequency]

        Time complexity: O(T), where T is the total number of characters in Dictionary
        Aux space complexity: O(T), since the primary data structure is the Trie
        """
        self.root = TrieNode("", None)
        for info in Dictionary:  # O(N), where N is the number of words
            word = info[0]
            definition = info[1]
            frequency = info[2]
            pointer = self.root

            for char in word:  # O(M), where M is the length of the word
                if frequency > pointer.highest_freq:
                    pointer.highest_freq = frequency
                    pointer.highest_data = info

                # if words have the same frequency
                elif frequency == pointer.highest_freq:
                    # compare words alphabetically and update the data accordingly
                    index = 0
                    if index < len(pointer.highest_data):
                        if word < pointer.highest_data[0]:
                            pointer.highest_data = info
                        elif word == pointer.highest_data[index]:
                            pointer.highest_data = info
                    else:
                        pointer.highest_data = info
                
                pointer.num_matches += 1
                pointer = pointer.get_child_node(char)

            # accounts for the prefix being its own word
            # it is updated in the outer loop, after the entire word is processed
            if frequency > pointer.highest_freq:
                pointer.highest_freq = frequency
                pointer.highest_data = info
            elif frequency == pointer.highest_freq:
                if pointer.highest_data and word < pointer.highest_data[0]:
                    pointer.highest_data = info
                
            pointer.data = info
            pointer.num_matches += 1


    def prefix_search(self, prefix):
        """
        Function description: Traverses the prefix trie to find the most frequent word that has the inputted prefix, then 
        returns the word, its definition and how many words contain that prefix in the dictionary file.

        Approach description: When a prefix is entered, the prefix trie is traversed to find the provided prefix. It 
        traverses the Trie by following the nodes corresponding to each character in the prefix. The search stops when the 
        prefix is fully matched or when no further nodes are available. The method then returns information about the most 
        frequent word with the matching prefix. There are multiple checks to account for empty string inputs, if there are no 
        existing matches for the prefix being searched, and if the prefix itself is the word with the highest frequency.

        Input:
            - prefix (str): a string input that is the prefix we want to search for

        Output: Returns a list that contains the most frequent word with the given prefix, its definition, and how many
        matches for the prefix was found. If no matches are found, then it returns [None, None, 0].

        Time complexity: O(M+N), where M is the length of the prefix and N is the total no. of characters in the most frequent
        word. In the best case, if the prefix is the most frequent word, then it is O(M+M) = O(2M) = O(M).
        Aux space complexity: O(M+N), where M is the length of the prefix and N is the total no. of characters in the word.
        """
        pointer = self.root
        for char in prefix:  # time complexity = O(len(prefix))
            pointer = pointer.node_search(char)
            # no child node so no matches found
            if pointer is None:
                return [None, None, 0]
            
            # if input is an empty string, return most frequent word in Dictionary
            if len(prefix) == 0:
                return pointer.highest_data
            
        # if prefix is also the most freq. word
        if pointer.highest_data and prefix == pointer.highest_data[0]:  # O(L) comparison, L = no. of char in the strings
            return [prefix, pointer.highest_data[1], pointer.num_matches]
        
        return [pointer.highest_data[0], pointer.highest_data[1], pointer.num_matches]        
    

if __name__ == "__main__":
    Dictionary = load_dictionary("Dictionary.txt")
    myTrie = Trie(Dictionary)
    # print(myTrie.prefix_search("ac"))


# 2. A Weekend Getaway
def allocate(preferences, licences):
    """
    Function description: This function allocates people to cars associated with a destination they prefer to travel to, or 
    return None if allocation is not possible. It checks whether allocation is possible or not by using the Ford-Fulkerson
    algorithm. The first round of allocations validates the constraint of each car having 2 drivers, and if it is met,
    the second round validates if everyone else can also be allocated.

    Approach description: A graph is created with nodes that represent people and cars, with a source and sink. Edges between 
    nodes represent a possible allocation of people to cars. In the first iteration, Ford-Fulkerson (FF) is used to check 
    if allocation of two drivers per car is possible as that is a constraint for this problem. If the allocation is possible, 
    a new graph is created to allocate everyone else and use FF again to check that allocation is possible while meeting the 
    constraints. By setting the appropriate capacities to account for the constraints of 2 drivers per car and 5 people total 
    per car, if the max. flow found by running FF is equal to how many people should be allocated, then allocation is possible.
    This is true because each car has an edge to the sink with capacity of 2 in the first iteration, and capacity of 3 in the
    second. These capacities restrict how many people can be allocated. The capacities on the edges between the source and
    people nodes is 1 to represent each person. By solving the max. flow problem, if it is equal to the number of people that
    need to be allocated, then allocation is possible.

    Input:
        - preferences (list of list): contains each person's preferences for which location they would like to go to
        - licences (list): contains people who have licences, each element in the list is a number representing the person

    Output: 
    Returns None if no allocation is possible because the constraints are not met, or if anyone has an empty preference list. 
    If allocation is possible, then it returns cars (a list of lists where each index corresponds to a car/destination and
    each list contains who is allocated to that car).

    Time complexity: O(n^3), where n is the number of people - (n people + n/5 cars can be simplified to n)
        - constructing a graph takes O(n^2) time and running FF takes O(n^3) time in the worst case.
    Aux space complexity: O(n^2), where n is the number of nodes for people and cars/destination in the graph
    """
    num_people = len(preferences)
    num_cars = math.ceil(num_people / 5)  # ceiling function for number of cars needed
    original_graph = construct_graph(num_people, num_cars)
    first_graph = [row[:] for row in original_graph]
    source = 0
    sink = 1

    # initially, everyone is set as a non-driver
    check_driver = ["non-driver"] * (2 + num_people + num_cars)
    check_driver[0] = "source"
    check_driver[1] = "sink"

    # show who has license to update role to driver
    cars = [[] for _ in range(num_cars)]  # aux. space: O(no. of cars)

    # create car nodes
    for car in range(num_cars):
        check_driver[car + 2 + num_people] = "car"
        first_graph[car + 2 + num_people][sink] = 2  # capacity is for 2 drivers first


    # if person has a license, they are labelled as a (potential) driver
    for person in licences:
        check_driver[person + 2] = "driver"

    # allocate drivers first
    flow = 0
    for person in range(num_people):
        index = person + 2
        if check_driver[index] == "driver":
            flow += 1
            # add edges between people and their preferences
            for dest in preferences[person]:
                first_graph[index][(dest + 2 + num_people)] = 1 
        first_graph[source][index] = 1 
        
    # if allocation of at least two drivers is not possible, then no allocation is possible
    drivers_allocation = FordFulkerson(first_graph, cars, check_driver)
    if drivers_allocation != num_cars*2:
        return None 

    # reset the graph back to its original state by making a copy
    second_graph = [row[:] for row in original_graph]
    flow = 0

    for car in range(num_cars):
        check_driver[car + 2 + num_people] = "car"
        second_graph[car + 2 + num_people][sink] = 3  # capacity is now 3 because 2 drivers already allocated

    # allocate everyone else who haven't been allocated in the first iteration
    for person in range(num_people):
        index = person + 2
        if check_driver[index] in ["non-driver", "driver"]:
            flow += 1
            for dest in preferences[person]:
                second_graph[index][(dest + 2 + num_people)] = 1
        second_graph[source][index] = 1

    remaining_allocation = FordFulkerson(second_graph, cars, check_driver)
    if remaining_allocation == num_people - num_cars*2:
        return cars
    else:
        return None


def construct_graph(num_people, num_cars):
    """
    Function description: Constructs a graph which is the size of the number of people + number of cars + 2 to account for
    the source and sink of the graph - graph is represented by an adjacency matrix.

    Input:
        - num_people (int): represents how many people need to be allocated for the trip
        - num_cars (int): represents how many cars will be needed

    Output: A graph with enough nodes to represent the source, sink, each person, and each car/destination.

    Time complexity: O((n+m)^2), where n is the number of people and m is the number of cars
    Aux space complexity: O((n+m)^2), where n is the number of people and m is the number of cars
    """
    # graph = [[0] * (num_people + num_cars + 2) * (num_people + num_cars + 2)]
    graph = [[0] * (num_people + num_cars + 2) for _ in range(num_people + num_cars + 2)]
    return graph


def find_augmenting_paths_bfs(graph, source, sink, paths):
    """
    Function description: An implentation of breadth-first search (BFS) which is used when running
    Ford-Fulkerson algorithm to traverse through the graph. In the worst case, it may visit every vertex and edge.

    Input:
        - graph: a graph that contains nodes representing the soruce, sink, people and cars
        - source: the source node where the search starts.
        - sink: the sink node, representing the endpoint of the path.
        - paths (list): used to track the paths from source to sink.

    Output: Returns True if an augmenting path from source to sink is found, False otherwise.

    Time complexity: O(V*E), where V is the number of vertices and E is the number of edges in the graph. 
    Aux space complexity: O(V), where V is the number of vertices.
    """
    visited = [False] * len(graph)
    queue = deque()
    queue.append(source)
    visited[source] = True

    while queue:
        current = queue.popleft()
        
        for next_node, capacity in enumerate(graph[current]):
            if not visited[next_node] and capacity > 0:
                queue.append(next_node)
                visited[next_node] = True
                paths[next_node] = current
                if next_node == sink:
                    return True
    return False


def FordFulkerson(graph, cars, check_driver):
    """
    Function description: This function implements the Ford-Fulkerson algorithm using a modified version of the Breadth-First 
    Search (BFS) to minimise costs (by minimising how many cars are needed). BFS is used to find augmenting paths in the 
    residual graph, and once a person is allocated to their car, their status will be updated. Changes to the car allocations
    are updated iteratively to achieve the most optimal allocations.
    In the worst case, this algorithm will run n^2 times, where n is the number of people. Each time it runs, it takes O(V*E)
    time for the modified BFS to run and search for an augmenting path, where V is number of nodes and E is number of edges.

    Input:
        - graph: an adjacency matrix representing a residual graph where each edge contains capacity and flow.
        - cars (list): stores which person is assigned to each car.
        - check_driver (list): indicates the driver status of nodes (car, allocated).

    Output: Returns the maximum flow in the network, which represents the optimal allocation to minimise costs.

    Time complexity: O(n^3), where n is the number of people (since there are n people nodes and n/5 car nodes). 
    Aux space complexity: O(n^2), where n is the number of people (n people nodes and n/5 car nodes).
    """
    paths = [-1]*len(graph)
    source = 0
    sink = 1
    max_flow = 0
    num_nodes = len(graph)

    for _ in range(num_nodes):
        current_flow = float("inf")  # set as inf to prevent creating unintentional constraints
        current_node = sink

        # no augmenting path found, terminate the loop
        if not find_augmenting_paths_bfs(graph, source, sink, paths):
            break  

        # update values of the residual graph by backtracking while checking who has been allocated 
        current_node = sink
        while (current_node != -1):
            while (current_node != source):
                parent_node = paths[current_node]

                # after a person is added to a car, they are marked as allocated
                if check_driver[current_node] == "car":
                    check_driver[paths[current_node]] = "allocated"
                    cars[len(cars) - num_nodes+ current_node].append(parent_node-2)

                # they can be re-allocated to another car to optimize how people are allocated to cars
                if check_driver[parent_node] == "car" and check_driver[current_node] == "allocated":
                    cars[len(cars) - num_nodes + parent_node].remove(current_node-2)

                current_flow = min(current_flow, graph[parent_node][current_node])
                current_node = paths[current_node]

            current_node = paths[current_node]

        max_flow += current_flow

        t_index = sink
        # forward edge increases capacity, backward edge decreases capacity
        while (t_index != source):
            s_index = paths[t_index]
            graph[s_index][t_index] -= current_flow
            graph[t_index][s_index] += current_flow
            t_index = paths[t_index]

    return max_flow


# test cases from instruction sheet/ed
# preferences = [[2], [0, 1, 2], [2, 0], [2, 1], [0], [0], [1, 0, 2], [0, 1, 2], [1], [1], [2, 0, 1], [2, 0, 1], [1]]
# licences = [0, 9, 5, 10, 1, 4, 3]
# print(allocate(preferences, licences))
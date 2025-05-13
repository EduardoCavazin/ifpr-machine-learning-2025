import json

def count_frequencies(items):
    frequencies = {}
    for item in items:
        key = f"{item} ({type(item).__name__})"
        frequencies[key] = frequencies.get(key, 0) + 1
    print(frequencies)


if __name__ == "__main__":
    example1 = [1, 2, 3, 2, 1, 2, 3, 1, 2]
    example2 = [1, 2, 3, 2, '1', 2, 3, 1, 2, 4, 4, 'mar', 'banana', 4, 'mar']
    exercise1 = [1, 2, 3, 3, 3, 4, 5, 5]
    exercise2 = ['apple', 'banana', 'apple', 'orange', 'banana']
    exercise3 = [True, False, True, False, True]
    exercise4 = [(1, 2), (2, 3), (1, 2), (3, 4), (1, 2)]
    exercise5 = [1, 'apple', True, 'banana', 1, 'banana']
    
    print("Exercise 1:")
    count_frequencies(exercise1)
    print("Exercise 2:")
    count_frequencies(exercise2)
    print("Exercise 3:")
    count_frequencies(exercise3)
    print("Exercise 4:")
    count_frequencies(exercise4)
    print("Exercise 5:")
    count_frequencies(exercise5)
    
    print("\nContage de nomes no JSON")
    
    with open('c:\\Codes\\Machine Learning\\Assets\\names.json', 'r') as file:
        data = json.load(file)
        count_frequencies(data['nomes-pessoas'])
    
    
    

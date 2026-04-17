"""This script has three bugs. Find and fix them all."""

def calculate_average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)  # Bug 1: division by zero if empty list


def find_longest_word(sentence):
    words = sentence.split(" ")
    longest = words[0]
    for word in words:
        if len(word) > len(longest):
            longest = words  # Bug 2: should be 'word' not 'words'
    return longest


def count_occurrences(text, target):
    count = 0
    for char in text:
        if char == target:
            count =+ 1  # Bug 3: should be += not =+
    return count


if __name__ == "__main__":
    print(calculate_average([10, 20, 30]))
    print(find_longest_word("the quick brown fox"))
    print(count_occurrences("hello world", "l"))

#!/bin/env python
import re
import json

def save_tokenized_vector(tokens, filename):
    with open(filename, "w") as f:
        json.dump(tokens, f)

def load_tokenized_vector(filename):
    with open(filename, "r") as f:
        tokens = json.load(f)
        return tokens

def tokenize_text(text):
    # Split the text into words and punctuation symbols using a regular expression
    # The pattern matches one or more characters that are either letters, digits, or apostrophes (for contractions),
    # or three consecutive dots (for ellipses), or any other non-whitespace character
    pattern = r"\w+|[\.\.\.]|\s|\S"
    tokens = re.findall(pattern, text)
    return tokens

def find_common_parts(word1, word2):
    # Find the longest common substring between the two words
    m = len(word1)
    n = len(word2)
    longest_common = ""
    for i in range(m):
        for j in range(n):
            k = 0
            while (i + k < m and j + k < n and word1[i + k] == word2[j + k]):
                k += 1
            if k > 2 and len(longest_common) < k:
                longest_common = word1[i:i+k]

    # If there is no common substring, return None
    if longest_common == "":
        return None

    # Find the non-common parts of the two words
    non_common1 = word1.split(longest_common)
    non_common2 = word2.split(longest_common)

    # Remove any empty strings from the non-common parts lists
    non_common1 = [part for part in non_common1 if part != ""]
    non_common2 = [part for part in non_common2 if part != ""]

    # Combine the longest common substring with the non-common parts
    result = [longest_common] + non_common1 + non_common2

    # Return the result
    return result


def find_subword(word, components):            
    for c in components:
        parts = find_common_parts(word, c)
        if parts:
            components.remove(c)
            for part in parts:
                components.add(part)
            return True
        
    return False


def create_word_components(words):
    wordCount = len(words) 
    print(f"Analyzing {wordCount} words...")
    components = set()
    count = 0
    for word in words:
        count += 1
        if count % int(wordCount / 10) == 0:
            print(f"{round(count / wordCount * 100)}%...")

        if word in components:
            continue
        if len(word) <= 2:
            components.add(word)
            continue
        
        found = find_subword(word, components)
        if not found:
            components.add(word)
    return components

with open("commedia.txt", "r") as dataFile:
    text = dataFile.read()

components = create_word_components(set(tokenize_text(text)))
save_tokenized_vector(sorted(components), "commedia_tokens.json")
print(f"Found {len(components)} tokens.")

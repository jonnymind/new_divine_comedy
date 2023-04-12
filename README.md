# new_divine_comedy
A playground to test AI GPT techniques, and build a new divine comedy.

The scripts analyze a text and create a model that imitate the original
text using randomized input.


## How to use

Install torch with 
```
pip3 install torch
```

### Step 1: Tokenize the source

* Tokenizer_char makes a character-wise tokenization of the target.
* Tokenizer_word breaks the source in word-components as large as possible.

Use:
```
python tokenizer[_char.py | _word.py] INPUT DICTIONARY TOKENIZED_OUTPUT
```

For example:
```
python tokenizer_char.py commedia.txt dict_char.txt commedia_tokenized_char.json
```

### Step 2: Train the model

```
python training.py DICTIONARY TOKENIZED_INPUT MODEL_OUTPUT
```

Example:
```
python training.py dict_char.json commedia_tokenized_char.json model_char.pth
```

### Step 3: Generate output

```
python generator.py DICTIONARY MODEL TOKEN_COUNT [RANDOM_SEED]
```

Example:
```
python generator.py dict_char.json model_char.pth 1000 12345
```

# Author

Giancarlo Niccolai. 

Algorithm based on this [GTP example](https://youtu.be/kCc8FmEb1nY) by Andrej Karpathy.

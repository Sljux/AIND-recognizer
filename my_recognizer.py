import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer

    for index in range(len(test_set.get_all_Xlengths())):
        x, lengths = test_set.get_item_Xlengths(index)

        probs = {}
        guess = (float('-inf'), None)

        for word, hmm in models.items():
            try:
                score = hmm.score(x, lengths)
            except Exception:
                score = float('-inf')

            probs[word] = score

            if score > guess[0]:
                guess = (score, word)

        probabilities.append(probs)
        guesses.append(guess[1])

    return probabilities, guesses

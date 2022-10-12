from math import exp 
from json import load

class GiberrishClassifier:
    def __init__(self, data_path:str='giberrish_model.json') -> None:
        data = load(open(data_path, 'r'))
        self.threshold = data["threshold"]
        self.log_probability_matrix = data["probabilities"]
        self.recognised_characters = data["characters"]
        
    def get_gibberish(self, text:str) -> str:
        return ''.join(filter(self.is_giberrish,text.split()))

    def remove_gibberish(self, text:str) -> str:
        return ''.join(filter(self.is_word,text.split()))
        
    def is_giberrish(self, word:str) -> bool:
        return not self.is_word(word)

    def is_word(self, word:str) -> bool:
        return self.average_transition_probability(word) > self.threshold

    def average_transition_probability(self, word:str) -> float:  
        word = self._normalise(word)
        character_bigrams = list(map(''.join,zip(word,word[1:])))        
        log_probability = sum(map(
            lambda bigram:self.log_probability_matrix[self.recognised_characters.index(bigram[0])][self.recognised_characters.index(bigram[1])],
            character_bigrams
        ))     
        n = len(character_bigrams) or 1
        return exp(log_probability / n)

    def _normalise(self, text:str) -> str:
        return ''.join(filter(
            lambda character:character in self.recognised_characters,
            text.lower()
        ))    

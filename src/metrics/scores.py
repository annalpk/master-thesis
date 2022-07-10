import numpy as np


class Similarity():
    """
        Computes the similarity measure between a predicted saliency map and the corresponding ground-truth map.
    """
    def __call__(self, input, label):
        input_sum = input.sum()
        label_sum = input.sum()
        input_norm = input / input_sum
        label_norm = label / label_sum
        min_array = np.minimum(input_norm, label_norm)
        return min_array.sum()


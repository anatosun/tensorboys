import numpy as np
from sklearn.preprocessing import minmax_scale
from scipy.spatial.distance import cdist
from pyts.metrics import dtw


def lower_contour(slice):

    index = np.where(slice <= 0.5)[0]

    if len(index) == 0:
        return 0
    else:
        return np.max(index)


def upper_contour(slice):

    index = np.where(slice <= 0.5)[0]

    if len(index) == 0:
        return len(slice)
    else:
        return np.min(index)


def fraction_black(slice):

    number = len(np.where(slice <= 0.5)[0])

    if number == 0:
        return 0.0
    else:
        return number/len(slice)


def fraction_black_between(slice):

    number = len(np.where(slice <= 0.5)[0])

    if number == 0:
        return 0.0
    else:

        index_min = np.min(np.where(slice <= 0.5))
        index_max = np.max(np.where(slice <= 0.5))

        return fraction_black(slice[index_min:index_max+1])


def transition_black_white(slice):

    counter = 0
    was_on_black = False

    for i in range(len(slice)):

        if slice[i] <= 0.5 and was_on_black == False:
            was_on_black = True

        elif slice[i] > 0.5 and was_on_black == True:
            was_on_black = False
            counter += 1

    return counter


# Compute the "time serie", the features for each slice of the image by sliding one pixel after the other
def compute_features_vector(image, normalize=True):

    rep = []
    for i in range(image.shape[1]):
        feature_vector = [lower_contour(image[:, i]), upper_contour(image[:, i]), fraction_black(
            image[:, i]), fraction_black_between(image[:, i]), transition_black_white(image[:, i])]
        rep.append(feature_vector)

    array = np.asarray(rep)

    if normalize:
        array = minmax_scale(array, feature_range=(0, 1), axis=0)

    return array


def compute_dtw(image1, image2, windows_size=0.5, normalize=True):

    feature_vector_1 = compute_features_vector(image1, normalize)
    feature_vector_2 = compute_features_vector(image2, normalize)

    dist_matrix = cdist(feature_vector_1, feature_vector_2)

    dtw_cost = dtw(precomputed_cost=dist_matrix, dist="precomputed",
                   method="sakoechiba", options={"window_size": windows_size})

    return dtw_cost

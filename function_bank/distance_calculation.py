
def euclidean(p: tuple, q: tuple) -> float:
    """
    Euclidean distance between two n-dimensional points.

    Formula:
        d(p, q) = sqrt( Σ (p_i - q_i)^2 )

    Parameters:
        p, q: Points with the same number of dimensions.

    Returns:
        float: Euclidean distance.
    """
    return sum((a - b) ** 2 for a, b in zip(p, q)) ** 0.5

def manhattan(p: tuple, q: tuple) -> float:
    """
    Manhattan distance between two n-dimensional points.

    Formula:
        d(p, q) = Σ |p_i - q_i|

    Parameters:
        p, q: Points with the same number of dimensions.

    Returns:
        float: Manhattan distance.
    """
    return sum(abs(a - b) for a, b in zip(p, q))

def chebyshev(p: tuple, q: tuple) -> float:
    """
    Chebyshev (Maximum Euclidean) distance between two n-dimensional points.

    Formula:
        d(p, q) = max(|p_i - q_i|)

    Parameters:
        p, q: Points with the same number of dimensions.

    Returns:
        float: Maximum Euclidean distance.
    """
    return max(abs(a - b) for a, b in zip(p, q))

def weighted_euclidean(p: tuple, q: tuple, weights: list) -> float:
    """
    Weighted Euclidean distance between two n-dimensional points.

    Formula:
        d(p, q) = sqrt( Σ w_i * (p_i - q_i)^2 )

    Parameters:
        p, q: Points with the same number of dimensions.
        weights: A list of weights for each dimension.

    Returns:
        float: Weighted Euclidean distance.
    """
    return sum(w * (a - b) ** 2 for w, a, b in zip(weights, p, q)) ** 0.5

def quadratic_form(p: tuple, q: tuple, M: list) -> float:
    """
    Quadratic form distance between two n-dimensional points.

    Formula:
        d(p, q) = (p - q)^T * M * (p - q)

    Parameters:
        p, q: Points with the same number of dimensions.
        M: A symmetric matrix of weights.

    Returns:
        float: Quadratic form distance.
    """
    diff = [a - b for a, b in zip(p, q)]
    return sum(a * b for a, b in zip(diff, [sum(M[i][j] * diff[j] for j in range(len(diff))) for i in range(len(diff))]))
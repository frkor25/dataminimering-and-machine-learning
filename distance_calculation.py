
def euclidean(p, q):
    return sum((a - b) ** 2 for a, b in zip(p, q)) ** 0.5

def manhattan(p, q):
    return sum(abs(a - b) for a, b in zip(p, q))

def maximum_euclidean(p, q):
    return max(abs(a - b) for a, b in zip(p, q))

def weighted_euclidean(p, q, weights):
    return sum(w * (a - b) ** 2 for w, a, b in zip(weights, p, q)) ** 0.5

def quadratic_form(p, q, M):
    diff = [a - b for a, b in zip(p, q)]
    return sum(a * b for a, b in zip(diff, [sum(M[i][j] * diff[j] for j in range(len(diff))) for i in range(len(diff))]))
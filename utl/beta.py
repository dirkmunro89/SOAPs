# estimate parameters of beta dist.
def getAlphaBeta(mu, sigma):
    alpha = mu**2 * ((1 - mu) / sigma**2 - 1 / mu)
    beta = alpha * (1 / mu - 1)
    return {"alpha": alpha, "beta": beta}
print(getAlphaBeta(0.1, 0.05))  # {alpha: 12, beta: 12}

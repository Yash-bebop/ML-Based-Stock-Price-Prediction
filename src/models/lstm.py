def create_sequences(X, y, lb):
    return (np.array([X[i-lb:i] for i in range(lb,len(X))]),
            np.array([y[i]        for i in range(lb,len(X))]))


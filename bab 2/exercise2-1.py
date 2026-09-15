def minarray(xs):
    m = xs[0]
    for x in xs:
        if m > x:
            m = x
    return m

data = [1, 2, 3, 4, 5, 6]
t = minarray(data)
print(t)
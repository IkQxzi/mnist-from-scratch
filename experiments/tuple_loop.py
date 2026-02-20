
def add(a, b):
    return a + b

def mult(a, b):
    return a * b

def div(a, b):
    return a / b

funcs = {}

for func in (add, mult, div):
    print(func)
    print(func(10, 2))

    funcs[func] = func(10, 2)

print(funcs)
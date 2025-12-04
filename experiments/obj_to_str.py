
class Calc():
    
    def __init__(self, a: int, b: int) -> None:
        self.a, self.b  = a, b
    
    def add(self):
        return self.a + self.b
    

calcs = [Calc(1, 2), Calc(2, 3)]

print(calcs)
print(str(calcs))

for obj in calcs:
    print(obj.__class__.__name__)
    print(obj.__class__.)
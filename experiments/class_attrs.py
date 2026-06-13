
class mycls: 

    def __init__(self, attr):
        self.attr = attr

class empty:

    def __init__(self):
        pass


myclass = mycls(1)
emptycls = empty()

print(myclass.__dict__)
if not emptycls.__dict__: 
    print(emptycls.__dict__)    
print(hasattr(myclass, 'attr'))
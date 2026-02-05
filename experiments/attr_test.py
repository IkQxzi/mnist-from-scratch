

class MyClass():

    def __init__(self) -> None:
        self.kwargs = {'class': self.__class__}



class1 = MyClass()

class2 = class1.kwargs['class']()

print(class2.kwargs)


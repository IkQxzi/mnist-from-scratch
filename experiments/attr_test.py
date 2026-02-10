

class MyClass():

    def __init__(self) -> None:
        self.kwargs = {'class': self.__class__}
        self.attr = None



class1 = MyClass()

class2 = class1.kwargs['class']()

class3 = MyClass(**class2.attr)

print(class2.kwargs)


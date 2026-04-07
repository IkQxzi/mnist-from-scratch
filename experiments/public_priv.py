

class MyClass:

    val = 0
    _val = 0
    
    def add_two_nums(self, a, b):
        return a + b
    
    def _add_two_nums(self, a, b):
        return a + b
    

cls = MyClass()

a, b = 1, 2

print(cls.add_two_nums(a, b))
print(cls._add_two_nums(a, b))

cls.val = 2
cls._val = 2

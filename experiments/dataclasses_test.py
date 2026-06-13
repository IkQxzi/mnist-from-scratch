from dataclasses import dataclass

@dataclass
class mycls:

    attr0: str
    attr1: str = '' 

myclass = mycls('the')
print(myclass)
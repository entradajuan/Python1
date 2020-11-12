class Person:

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def getoneyear(self):
        self.age +=1

    def sayHello(self):
        return "Hello my name is ", self.name

    def __call__(self, x, y):
        return x + y

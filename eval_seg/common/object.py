class Object:

    def __init__(self, name=None):
        self.name = name or self.__class__.__name__

    def __str__(self):
        return '<' + self.name + '> ' + str(self.__dict__)

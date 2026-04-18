class Robot:
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def move(self, direction):
        print(f"{self.name} is moving {direction}.")

    def speak(self, message):
        print(f"{self.name} says: {message}")
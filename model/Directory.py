class Directory:
    def __init__(self, path):
        self.path = path
        self.subdirectories = []

    def add_subdirectory(self, subdirectory):
        self.subdirectories.append(subdirectory)

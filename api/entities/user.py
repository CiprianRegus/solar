import sys

sys.path.insert("..", 0)


class User:
    
    def __init__(self, username, password):
        
        self.username = username
        self.password = password

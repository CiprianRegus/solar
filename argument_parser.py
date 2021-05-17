class Arg:
    
    def __init__(self, name, nvalues, arg_type, function):
        """
            name: the name of the argument
            
            nvalues: the number of values which will follow the arg string. By providing -1, the parser will match 
            any number of values of type arg_type
            
            arg_type: the type of argument values. Can either be int, float or string.
            
            function: this function will be called with the matched arguments as a list: function(arg, [arg_type])
        """
        if arg_type not in ["string", "int", "float"]:
            raise TypeError("Invalid type provided (only int, float, string accepted)")
        if nvalues < 0:
            raise ValueError("Invalid number of argument values: ", nvalues)
        if not (name.startswith("-") and len(name) == 2) and not (name.startswith("--") and len(name) > 2):
            raise ValueError("Invalid argument name: ", name)
        
        self.name = name
        self.nvalues = nvalues
        self.arg_type = arg_type
        self.function = function
        
    def __call__(self, arg_values):
        arg_values_converted= []
        for e in arg_values:
            try:
                if self.arg_type == "string":
                    arg_values_converted.append(e)
                elif self.arg_type == "int":
                    arg_values_converted.append(int(e))
                elif self.arg_type == "float":
                    arg_values_converted.append(float(e))
            except:
                raise TypeError("Cannot convert to type: ", self.arg_type, " from ", type(arg_values))
        return self.function(self.name, arg_values_converted)

class ArgParser:
    
    def __init__(self, input_args):
        """
            input_args: a list of strings representing the program arguments, similar to sys.argv
        """
        self.input_args = input_args
        self.args = {}
         
    def add_arg(self, name, nvalues, arg_type, function):
        """
            arg: object of type Arg        
        """
        arg = Arg(name, nvalues, arg_type, function)
        self.args[arg.name] = arg
    
    def __call__(self):
        for i, e in enumerate(self.input_args):
            if e.startswith("-"):
                current_arg = self.args[e]
                values = self.input_args[i + 1: i + current_arg.nvalues + 1]
                print("Calling ", current_arg.name)
                current_arg(values) 
      
    def print_args(self):
        for e in self.args.keys():
            print(e)  
    
        
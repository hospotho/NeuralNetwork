from math import tanh, log, e
import random


def ReLU(x):
    return x if x > 0 else 0


def der_ReLU(x):
    return 1 if x > 0 else 0


def LeakReLU(x):
    return x if x > 0 else 0.01*x


def der_LeakReLU(x):
    return 1 if x > 0 else 0.01


def Mish(x):
    return x * tanh(log(1 + e**x))


def der_Mish(x):
    return 1/(((e ** x + 1) ** 2 + 1) ** 2) * (e ** x) * (e**(3*x) + 4 * e**(2*x) + (6 + 4*x) * e**x + 4 + 4*x)


activation_function = {"ReLU": ReLU, "LeakReLu": LeakReLU, "Mish": Mish}
activation_function_derivative = {"ReLU": der_ReLU, "LeakReLu": der_LeakReLU, "Mish": der_Mish}


class node():
    def __init__(self, connection, function="LeakReLu"):
        self.net = []
        self.weight = [random.random()*2-1 for _ in range(connection)]
        self.bias = random.random()*2-1
        self.output = 0
        self.BPnet = []
        self.error = 0
        self.function = float if function == None else activation_function[function]
        self.der_function = bool if function == None else activation_function_derivative[function]

    def forward(self, input):
        self.net.append(input)

    def calculate(self):
        temp = 0
        for i in range(len(self.weight)):
            temp = temp + self.net[i] * self.weight[i]
        temp = temp + self.bias
        self.output = self.function(temp)
        self.net = []

    def calculate_error(self):
        temp = 0
        for error in self.BPnet:
            temp = temp + error
        self.error = temp
        self.BPnet = []


class Network():
    def __init__(self):
        self.input_node = 0
        self.layers = []
        self.last_layer_connection = 0
        self.output = []
        self.train_rate = 0.05

    def info(self):
        for layer in self.layers:
            print([(node.weight, node.bias) for node in layer])

    def add_input_layer(self, needed_node):
        self.input_node = needed_node
        self.last_layer_connection = needed_node

    def add_hidden_layer(self, needed_node):
        if self.last_layer_connection == -1:
            print("Output layer already exists")
            return -1
        temp = []
        for i in range(needed_node):
            new_node = node(connection=self.last_layer_connection)
            temp.append(new_node)
        self.layers.append(temp)
        self.last_layer_connection = needed_node

    def add_output_layer(self, needed_node):
        if self.last_layer_connection == -1:
            print("Output layer already exists")
            return -1
        temp = []
        for i in range(needed_node):
            no_function_node = node(connection=self.last_layer_connection, function=None)
            temp.append(no_function_node)
        self.layers.append(temp)
        self.last_layer_connection = -1

    def predict(self, input_list):
        temp = input_list
        for layer in self.layers:
            for node in layer:
                for input in temp:
                    node.forward(input)
                node.calculate()
            temp = []
            temp.extend([node.output for node in layer])
        self.output = temp

    def training(self, input, output):
        self.predict(input)
        errors = []
        count = 0
        for node in self.layers[-1]:
            node.error = output[count]-self.output[count]
            errors.append(node.error)
            count = count+1
        count = 0
        print(errors)
        for i in range(len(self.layers)-1, 0, -1):
            for down_node in self.layers[i]:
                for high_node in self.layers[i-1]:
                    for error in errors:
                        high_node.BPnet.append(error*down_node.weight[count])
                        count = count + 1
                    high_node.calculate_error()
                    count = 0
            errors = []
            errors.extend([node.error for node in self.layers[i-1]])
        temp = input
        for layer in self.layers:
            for node in layer:
                for i in range(len(input)):
                    node.weight[i] = node.weight[i]+self.train_rate*node.der_function(node.output)*node.error*temp[i]
                node.bias = node.bias+self.train_rate*node.der_function(node.output)*node.error
                node.error = 0
            temp = []
            temp.extend([node.output for node in layer])


if __name__ == "__main__":
    n = Network()
    n.add_input_layer(2)
    n.add_hidden_layer(5)
    n.add_hidden_layer(4)
    n.add_output_layer(1)

    for i in range(10000):
        k = random.random()
        training_input = [k,k+5]
        training_output = [k+2.5]
        n.training(training_input, training_output)

    test_input_set = [[i,i+5] for i in range(4)]
    for input in test_input_set:
        n.predict(input)
        print("Input:", end=' ')
        print(input)
        print("Output:", end=' ')
        print(n.output)

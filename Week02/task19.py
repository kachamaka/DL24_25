import numpy as np
import graphviz
import math

class Value:
    def __init__(self, data: float, prev: set = set(), op: str = "", label: str = "") -> None:
        self.data = data
        self._prev = prev
        self._op = op
        self._grad = 0.0 
        self._label = label
        self._backward = lambda: None

    def __add__(self, other, op="+") -> "Value":
        if not isinstance(other, Value):
            result = Value(self.data + other, {self}, op=op)
            
            def backward():
                self._grad += result._grad
            
            result._backward = backward
            return result

        result = Value(self.data + other.data, {self, other}, op=op)
        
        def _backward():
            self._grad += result._grad
            other._grad += result._grad
        
        result._backward = _backward
        return result
    
    def __radd__(self, other) -> "Value":
        return self.__add__(other)

    def __mul__(self, other, op="*") -> "Value":
        if not isinstance(other, Value):
            result = Value(self.data * other, {self}, op=op)

            def backward():
                self._grad += other * result._grad

            result._backward = backward
            return result
        
        result = Value(self.data * other.data, {self, other}, op=op)

        def _backward():
            self._grad += other.data * result._grad
            other._grad += self.data * result._grad

        result._backward = _backward
        return result
    
    def __rmul__(self, other) -> "Value":
        return self.__mul__(other)
    
    def __truediv__(self, other, op="/") -> "Value":
        op = "**-1" if self.data == 1 else op
        if not isinstance(other, Value):
            result = Value(self.data / other, {self}, op=op)

            def backward():
                self._grad += (1/other) * result._grad

            result._backward = backward
            return result
        
        result = Value(self.data / other.data, {self, other}, op=op)

        def _backward():
            self._grad += (1/other.data) * result._grad
            other._grad += (-self.data/np.power(other.data, 2)) * result._grad

        result._backward = _backward
        return result
    
    def __rtruediv__(self, other, op="/") -> "Value":
        if not isinstance(other, Value):
            result = Value(other / self.data, {self}, op=op)

            def backward():
                self._grad += (-other/np.power(self.data, 2)) * result._grad

            result._backward = backward
            return result
        
        result = Value(other.data / self.data, {self, other}, op=op)

        def _backward():
            other._grad += (1/self.data) * result._grad
            self._grad += (-other.data/np.power(self.data, 2)) * result._grad

        result._backward = _backward
        return result

    def __pow__(self, other, op="**") -> "Value":
        if not isinstance(other, Value):
            result = Value(np.power(self.data, other), {self}, op=op)

            def backward():
                self._grad += (other * np.power(self.data, (other - 1))) * result._grad

            result._backward = backward
            return result

        result = Value(np.power(self.data, other.data), {self, other}, op=op)

        def _backward():
            self._grad += (other.data * np.power(self.data, (other.data - 1))) * result._grad
            other._grad += np.power(self.data, other.data) * math.log(self.data) * result._grad

        result._backward = _backward
        return result
    
    def __rpow__(self, other, op="**") -> "Value":
        if not isinstance(other, Value):
            result = Value(np.power(other, self.data), {self}, op=op)

            def backward():
                self._grad += (other * np.power(self.data, (other - 1))) * result._grad

            result._backward = backward
            return result

        result = Value(np.power(self.data, other.data), {self, other}, op=op)

        def _backward():
            other._grad += (self.data * np.power(other.data, (self.data - 1))) * result._grad
            self._grad += np.power(other.data, self.data) * math.log(other.data) * result._grad

        result._backward = _backward
        return result
    
    def exp(self, op="e") -> "Value":
        e = math.e
        result = Value(np.exp(self.data), {self}, op=op)

        def backward():
            self._grad += np.exp(self.data) * result._grad

        result._backward = backward
        return result

    def tanh(self) -> "Value":
        tanh_data = math.tanh(self.data)
        result = Value(tanh_data, {self}, op="tanh")

        def _backward():
            self._grad += (1 - np.power(tanh_data, 2)) * result._grad

        result._backward = _backward
        return result

    def __repr__(self) -> str:
        return f"Value(data={self.data}, label={self._label})"


    def set_label(self, label: str) -> None:
        self._label = label

    def backward(self) -> None:
        topo_order = top_sort([self])
        self._grad = 1.0

        for node in topo_order:
            node._backward()

def extract_parents(value: "Value", nodes: set["Value"], edges: set[tuple["Value", "Value"]]) -> None:
    if value in nodes:
        return

    nodes.add(value)
    if not value._prev:
        return

    for prev in value._prev:
        edges.add((prev, value))
        extract_parents(prev, nodes, edges)

def trace(value: "Value") -> tuple[set["Value"], set[tuple["Value", "Value"]]]: 
    nodes, edges = set(), set()
    extract_parents(value, nodes, edges)
    return nodes, edges

def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(filename='01_result', format='svg', graph_attr={'rankdir': 'LR'}) 

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))

        dot.node(name=uid, label=f'{{ {n._label} | data: {n.data:.4f} | grad: {n._grad:.4f} }}', shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

def top_sort(values: list[Value]) -> list[Value]:
    visited = set()
    result = []

    def dfs(node: Value):
        if node in visited:
            return

        visited.add(node)
        for parent in node._prev:
            dfs(parent)

        result.append(node)

    for value in values:
        dfs(value)

    return result[::-1] 

def main() -> None:
    x1 = Value(data=2.0, label="x1")
    x2 = Value(data=0.0, label="x2")
    w1 = Value(data=-3.0, label="w1")
    w2 = Value(data=1.0, label="w2")
    b = Value(data=6.8813735870195432, label="b")
    two = Value(data=2.0, label="2")
    one = Value(data=1.0, label="1")
    minus_one = Value(data=-1.0, label="-1")

    x1w1 = x1 * w1
    x1w1.set_label(label="x1w1")
    x2w2 = x2 * w2
    x2w2.set_label(label="x2w2")
    x1w1_plus_x2w2 = x1w1 + x2w2

    logit = x1w1_plus_x2w2 + b
    logit.set_label(label="logit")

    logit_times_two = logit * two

    e = logit_times_two.exp()
    e.set_label(label="e")

    e_plus_one = e + one
    e_plus_one_reciprocate = e_plus_one ** (-1)

    e_minus_one = e + minus_one

    L = e_minus_one * e_plus_one_reciprocate
    L.set_label(label="L")

    L.backward()

    draw_dot(L).render(directory='./graphviz_output', view=True)

if __name__ == "__main__":
    main()

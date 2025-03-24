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
            self._grad += (e * np.power(self.data, (e - 1))) * result._grad

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
    x = Value(2.0, label='x')

    expected = Value(4.0)

    actuals = {
        'actual_sum_l': x + 2.0,
        'actual_sum_r': 2.0 + x,
        'actual_mul_l': x * 2.0,
        'actual_mul_r': 2.0 * x,
        'actual_div_r': (x + 6.0) / 2.0,
        'actual_pow_l': x**2,
        'actual_exp_e': x**2,
    }

    assert x.exp().data == np.exp(2), f"Mismatch for exponentiating Euler's number: expected {np.exp(2)}, but got {x.exp().data}."

    for actual_name, actual_value in actuals.items():
        assert actual_value.data == expected.data, f'Mismatch for {actual_name}: expected {expected.data}, but got {actual_value.data}.'

    print('All tests passed!')

if __name__ == "__main__":
    main()

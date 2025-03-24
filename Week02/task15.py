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

    def __add__(self, other: "Value") -> "Value":
        result = Value(self.data + other.data, {self, other}, op="+")
        
        def _backward():
            self._grad = result._grad
            other._grad = result._grad
        
        result._backward = _backward
        return result

    def __mul__(self, other: "Value") -> "Value":
        return Value(self.data * other.data, {self, other}, op="*")

    def __repr__(self) -> str:
        return f"Value(data={self.data}, label={self._label})"
    
    def tanh(self) -> "Value":
        tanh_data = math.tanh(self.data)
        return Value(tanh_data, {self}, op="tanh")

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
    x = Value(10.0, label="x")
    y = Value(5.0, label="y")
    z = x + y
    z.set_label(label="z")

    z.backward()

    draw_dot(z).render(directory='./graphviz_output', view=True)


if __name__ == "__main__":
    main()

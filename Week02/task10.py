import graphviz

class Value:
    def __init__(self, data: float, prev: set = set(), op: str = "", label: str = "") -> None:
        self.data = data
        self._prev = prev
        self._op = op
        self._grad = 0.0 
        self._label = label

    def __add__(self, other: "Value") -> "Value":
        return Value(self.data + other.data, {self, other}, op="+")

    def __mul__(self, other: "Value") -> "Value":
        return Value(self.data * other.data, {self, other}, op="*")

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def set_label(self, label: str) -> None:
        self._label = label


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
    dot = graphviz.Digraph(filename='01_result', format='svg', graph_attr={
                           'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node

        dot.node(name=uid, label=f'{{ {n._label} | data: {n.data:.4f} | grad: {n._grad:.4f} }}', shape='record')
        if n._op:
            # if this value is a result of some operation, create an "op" node for the operation
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to the node of the operation
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the "op" node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

def manual_der(a, b, c, d, e, f, L):
    # d(L)/d(L) = 1
    L._grad = 1.0 

    # d(L)/d(d) = d(d * f)/d(d) = f.data (-2)
    d._grad = f.data * L._grad 

    # d(L)/d(f) = d(d * f)/d(f) = d.data (4)
    f._grad = d.data * L._grad 
    
    # d(d)/d(e) = d(e + c)/d(e) = 1
    # d(L)/d(e) = d(L)/d(d) * d(d)/d(e) = d._grad (-2)
    e._grad = d._grad 

    # d(d)/d(c) = d(e + c)/d(c) = 1
    # d(L)/d(c) = d(L)/d(d) * d(d)/d(c) = d._grad (-2)
    c._grad = d._grad 

    # d(e)/d(a) = d(a * b)/d(a) = b.data (2)
    # d(L)/d(a) = d(L)/d(e) * d(e)/d(a) = e._grad * b.data = (-2) * (2) = (-4)
    a._grad = b.data * e._grad 
    
    # d(e)/d(b) = d(a * b)/d(b) = a.data (-3)
    # d(L)/d(b) = d(L)/d(e) * d(e)/d(b) = e._grad * a.data = (-2) * (-3) = (6)
    b._grad = a.data * e._grad


def main() -> None:
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    e = a * b
    e.set_label(label="e")
    d = e + c
    d.set_label(label="d")
    f = Value(-2.0, label="f")
    L = d * f
    L.set_label("L")

    manual_der(a, b, c, d, e, f, L)

    # This will create a new directory and store the output file there.
    # With "view=True" it'll automatically display the saved file.
    draw_dot(L).render(directory='./graphviz_output', view=True)


if __name__ == "__main__":
    main()

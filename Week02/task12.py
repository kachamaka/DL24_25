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

def main() -> None:
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    b = Value(6.7, label="b")
    
    x1w1 = x1 * w1
    x1w1.set_label("x1w1")
    
    x2w2 = x2 * w2
    x2w2.set_label("x2w2")
    
    x1w1_plus_x2w2 = x1w1 + x2w2
    x1w1_plus_x2w2.set_label("x1w1 + x2w2")
    
    logit = x1w1_plus_x2w2 + b
    logit.set_label("logit")
    
    draw_dot(logit).render(directory='./graphviz_output', view=True)


if __name__ == "__main__":
    main()

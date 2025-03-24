class Value:
    def __init__(self, data: float, prev: set = set(), op: str = "") -> None:
        self.data = data
        self._prev = prev
        self._op = op
 
    def __add__(self, other: "Value") -> "Value":
        return Value(self.data + other.data, {self, other}, op="+")

    def __mul__(self, other: "Value") -> "Value":
        return Value(self.data * other.data, {self, other}, op="*")

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    

def extract_parents(value: "Value", nodes: set["Value"], edges: set[tuple["Value", "Value"]]) -> None:
    if value not in nodes:
        nodes.add(value)
        if value._prev:
            for prev in value._prev:
                edges.add((prev, value))
                extract_parents(prev, nodes, edges)

def trace(value: "Value") -> tuple[set["Value"], set[tuple["Value", "Value"]]]:
    nodes, edges = set(), set()
    extract_parents(value, nodes, edges)
    return nodes, edges


def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    
    nodes, edges = trace(x)
    print('x')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(y)
    print('y')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(z)
    print('z')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(result)
    print('result')
    print(f'{nodes=}')
    print(f'{edges=}')


if __name__ == "__main__":
    main()

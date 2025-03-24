class Value:
    def __init__(self, data: float, prev: set = None, op: str = "") -> None:
        self.data = data
        self._prev = prev
        self._op = op

    def __add__(self, other: "Value") -> "Value":
        return Value(self.data + other.data, {self, other}, op="+")

    def __mul__(self, other: "Value") -> "Value":
        return Value(self.data * other.data, {self, other}, op="*")

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result._op)


if __name__ == "__main__":
    main()

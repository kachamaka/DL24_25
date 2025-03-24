class Value:
    def __init__(self, data: float, _prev: set = None) -> None:
        self.data = data
        self._prev = _prev

    def __add__(self, other: "Value") -> "Value":
        return Value(self.data + other.data, {self, other})

    def __mul__(self, other: "Value") -> "Value":
        return Value(self.data * other.data, {self, other})

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result._prev)


if __name__ == "__main__":
    main()

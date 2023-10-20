class KGEntity:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name
    

class KGProperty:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name


class Triple: 
    def __init__(self, head: KGEntity, rel: KGProperty, tail: KGEntity) -> None:
        self.head = head
        self.rel = rel
        self.tail = tail

    def __str__(self) -> str:
        return f"({self.head}, {self.rel}, {self.tail})"
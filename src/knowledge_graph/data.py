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
    def __init__(self, head_entity: KGEntity, relation: KGProperty, tail_entity: KGEntity) -> None:
        self.head_entity = head_entity
        self.relation = relation
        self.tail_entity = tail_entity

    def __str__(self) -> str:
        return f"({self.head_entity}, {self.relation}, {self.tail_entity})"
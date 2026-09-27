"""Each instance as one record of its sheet."""

__all__ = ["fields"]

from living_memory import _field, _order, _relation
from living_memory._relation import Placed


def fields(placed: Placed) -> list[str]:
    # An instance is one record: its keys, then the text of each column;
    # a column whose value is written elsewhere, or absent, is empty
    # (Codd, 1.3. A Relational View of Data; spec › record.1, field.1,
    # field.2, relation.17).
    one = _relation.relation(placed)
    held = _relation.content(placed)
    written = []
    for column in _order.columns(_relation.keys(one), _relation.domains(one)):
        if isinstance(column, str):
            written.append(_relation.key(placed, column))
        elif column not in held:
            written.append("")
        elif column.role == "type":
            written.append(_field.type(held[column]))
        elif column.role == "run":
            written.append(held[column])
        else:
            written.append(_field.text(held[column]))
    return written

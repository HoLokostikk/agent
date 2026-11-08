def transpose(m: list[list[int]]) -> list[list[int]]:
    """Transpose a rectangular matrix (rows x cols -> cols x rows)."""
    return [[m[j][i] for j in range(len(m))] for i in range(len(m))]

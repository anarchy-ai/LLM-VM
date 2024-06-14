from parser import (
    Elements,
    Element,
    Vec,
    Vectors,
    )

"""Module corresponds to some utilities that enable the interpretation
of linear algebra expressions by converting the matrix and vector AST nodes
into Python arrays."""

def get_vector_elements(ast):
    """given an AST it takes the elements of a vector and puts them in a list."""
    elements = []
    while ast is not None:
        if isinstance(ast, Elements):
            elements.append(ast.e1)
            ast = ast.elements

        else:
            elements.append(ast.exp)
            ast = None

    return elements

def get_matrix_elements(ast):
    """given an AST take the elements and put them on a 2D list."""
    matrix = []

    if isinstance(ast, Vectors):
       matrix.extend(get_matrix_elements(ast.exp))
       matrix.extend(get_matrix_elements(ast.expressions))


    elif isinstance(ast, Elements):
        vector_entries = get_vector_elements(ast)
        matrix.append(vector_entries)

    elif isinstance(ast, Vec):
        vector_entries = get_vector_elements(ast.elements)
        matrix.append(vector_entries)

    else:
        print("no more instances")

    return matrix

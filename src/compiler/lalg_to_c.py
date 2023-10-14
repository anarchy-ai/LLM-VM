from ast_to_lalg import (
    LalgInt,
    LalgVec,
    LalgMatrix,
    LalgOp,
    LalgExps,
    LalgForLoop,
    LalgForLoop2D,
    )

def lalg_to_c(lalg_ast):
    match lalg_ast:
        case LalgForLoop(LalgExps(exps), LalgInt(n), LalgInt(i), LalgOp(op)):
            if op == '+':
                if all(isinstance(x, LalgVec) for x in exps):
                    vec = exps[0].vec
                    vec2 = exps[1].vec
                    return generate_c_for_vec_addition(vec, vec2, n, i, op)
                    
            else:
                raise ValueError('{} is not a valid 1D for loop expression.'.format(lalg_ast))
        case LalgForLoop2D(LalgExps(exps), LalgInt(n), LalgInt(inner_n), LalgInt(i), LalgInt(j), LalgOp(op)):
            if op == '+':
                if all(isinstance(x, LalgMatrix) for x in exps):
                    matrix = exps[0].matrix
                    matrix2 = exps[1].matrix

                    return generate_c_for_matrix_addition(matrix, matrix2, n, inner_n, i, j)

                else:
                    raise ValueError('{} is not a valid 2d for loop expression.'.format(lalg_ast))

        case _:
            raise ValueError('{} is not valid LALG ast.'.format(lalg_ast))


def generate_c_for_vec_addition(vec, vec2, n, i, op):
    exp = f"""int vec[] = {{{generate_c_for_vector(vec)}}};
    int vec2[] = {{{generate_c_for_vector(vec2)}}};

    vector *v = initialize_vector(vec, {n});

    vector *v2 = initialize_vector(vec2, {n});

    add_vectors(v, v2);"""

    return exp

def generate_c_for_matrix_addition(matrix, matrix2, n, inner_n, i, j):
    exp = f"""int matrix1[{n},{inner_n}] = {{{generate_c_for_matrix(matrix)}}};
    int matrix2[{n}, {inner_n}] = {{{generate_c_for_matrix(matrix2)}}};
    
    matrix *mat = initialize_matrix(matrix1, {n}, {inner_n});
    
    matrix *mat2 = initialize_matrix(matrix2, {n}, {inner_n});

    add_matrices(mat, mat2);"""

    return exp
    
    

def generate_c_for_vector(vec):
    return ', '.join(map(str, vec))
    

def generate_c_for_matrix(matrix):
    def generate_vec(vec):
        return f"{{{', '.join(map(str, vec))}}}"

    return ', '.join(map(generate_vec, matrix))
                             

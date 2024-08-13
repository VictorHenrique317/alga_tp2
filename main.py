import numpy as np
import math
import pickle

from functools import reduce
from collections import defaultdict

# Lista de primos para o crivo quadrático
FIRST_PRIMES: list[int] = list()


def powMod(a: int, k: int, M: int) -> int:
    """
    Calcula a^k mod M em tempo log(k).
    """

    # No negative powers
    if k < 0:
        return 0

    rem: int = 1

    # For every bit in k, square 'rem', mult by 'a' if bit is 1
    for bit in bin(k)[2:]:
        rem **= 2

        if bit == "1":
            rem *= a

        rem = rem % M

    return rem


def legendre_symbol(a: int, p: int) -> int:
    """
    Calcula o Símbolo de Legendre (a/p)
    """

    ls = powMod(a, (p - 1) // 2, p)

    return -1 if ls == p - 1 else ls


def tonelli_shank(n: int, p: int) -> int | None:
    """
    Encontra uma raiz quadrada de n módulo p, se existir.
    """

    # Verifica se n é resíduo quadrático
    if legendre_symbol(n, p) != 1:
        return None

    # Encontra q e s tais que p - 1 = q * 2^s com q ímpar
    s = 0
    q = p - 1
    while q % 2 == 0:
        s += 1
        q //= 2

    # Encontra um não-resíduo quadrático z
    z = 2
    while legendre_symbol(z, p) != -1:
        z += 1

    # Inicializa variáveis
    m = s
    c = powMod(z, q, p)
    t = powMod(n, q, p)
    r = powMod(n, (q + 1) // 2, p)

    while t != 1:
        # Encontra o menor i tal que t^(2^i) = 1 (mod p)
        i = 1
        temp = (t * t) % p
        while temp != 1:
            temp = (temp * temp) % p
            i += 1

        # Atualiza b, t, r
        b = powMod(c, 2 ** (m - i - 1), p)
        m = i
        c = (b * b) % p
        t = (t * c) % p
        r = (r * b) % p

    # First root is r, second is -r mod p
    return (r, -r % p)


def sieve(no: int, p: int) -> int:
    """
    Divide 'no' por 'p' enquanto for possível (criva de divisão).

    Retorna o resultado da divisão.
    """

    while no % p == 0:
        no //= p

    return no


def approximate_B_value(n: int) -> int:
    """
    Calcula um valor aproximado para B para uso no crivo quadrático, baseado no tamanho de N.
    """

    l = math.exp(math.sqrt(math.log(n) * math.log(math.log(n))))

    b = math.pow(l, 1 / math.sqrt(2))

    return math.ceil(b) + 1


def build_factor_base(n: int, b: int) -> list[int]:
    """
    Constrói a base de fatores para o crivo quadrático.

    Considera somente os que podem aparecer na fatoração dos candidatos a B-smooth.
    """

    factor_base: list[int] = [2]

    for p in FIRST_PRIMES[1:]:
        if p > b:
            break

        # Check if p is a quadratic residue (Euler's Criteria)
        # Skip if not
        if powMod(n, (p - 1) // 2, p) != 1:
            continue

        # Add to factor base if n is a quadratic residue
        factor_base.append(p)

    return factor_base


def get_factor_base_roots(n: int, factor_base: list[int]) -> dict[int, tuple[int, int]]:
    """
    Pré-computa as raízes quadradas de n módulo cada primo da base de fatores.

    Utiliza o algoritmo de Tonelli-Shanks.
    """

    roots: dict[int, tuple[int, int]] = dict()

    for f in factor_base:
        # 2 will be tested manually later
        if f == 2:
            continue

        # Get roots
        r1, r2 = tonelli_shank(n, f)

        roots[f] = (r1, r2)

    return roots


def build_list_to_sieve(
    offset: int, n: int, a: int, b: int, fb_roots: dict[int, tuple[int, int]]
) -> tuple[list[int], dict[int, list[int]]]:
    """
    Constrói a lista de polinômios t^2 - n para serem crivados.

    Também pré-computa os divisores de cada polinômio, para evitar divisões desnecessárias.
    """

    ls: list[int] = list()
    ls_divisors: dict[int, list[int]] = defaultdict(list)

    for i in range(b):
        # Calculate t, we can offset its starting point
        t = a + i + offset * b

        # Calculate t^2 - n polynomial
        s = t**2 - n

        # Add to sieve list
        ls.append(s)

        # Add factors to division list
        # Test 2 manually
        if s % 2 == 0:
            ls_divisors[s].append(2)

    # Test other factors
    for f, (r1, r2) in fb_roots.items():
        idx1: int = (r1 - (a + offset * b)) % f
        idx2: int = (r2 - (a + offset * b)) % f

        # Build lookup for ls divisors
        while idx1 < len(ls):
            ls_divisors[ls[idx1]].append(f)
            idx1 += f

        while idx2 < len(ls):
            ls_divisors[ls[idx2]].append(f)
            idx2 += f

    return (ls, ls_divisors)


def get_matrix_kernel_solutions(matrix: np.ndarray) -> list[np.ndarray]:
    """
    Calcula as soluções para o sistema de equações xA = 0 em Zmod2.

    Em específico, calcula as bases do kernel da matriz utilizando eliminação de Gauss.

    Complexidade: O(m * n) onde m é o número de linhas e n é o número de colunas.

    Baseado em: https://en.wikipedia.org/wiki/Kernel_%28linear_algebra%29#Computation_by_Gaussian_elimination
    """

    # We are working with Zmod2, this simplifies some operations
    # We also need to transpose the matrix since we are calculating the left kernel (xA = 0)
    M = np.mod(matrix.T, 2)

    rows, cols = M.shape

    # We will augment the matrix with the identity matrix as per the algorithm from the wiki
    I = np.eye(cols, dtype=int)
    A = np.concatenate((M, I), axis=0)

    # We need to transpose the matrix to so we can get its row echeleon form instead of the column echeleon form
    # They are equivalent. We will save the original dimensions to know where the identity matrix ends
    A = A.T
    cols_e = rows
    rows_e = cols

    # Get row echeleon form for the first cols_e columns
    for i in range(cols_e):
        # Find a row with a 1 in the i-th column
        for j in range(i, rows_e):
            if A[j][i] == 1:
                A[i], A[j] = A[j].copy(), A[i].copy()
                break

        # Eliminate the 1s in the i-th column (we are working with Zmod2 so -1 is 1 and we don't need to divide/multiply)
        for j in range(i + 1, rows_e):
            if A[j][i] == 1:
                A[j] = np.mod((A[j] + A[i]), 2)

    solutions = []

    # Basis are the rows in the former identity matrix where the row in M is all 0
    for r in range(rows_e):
        # If row in M is 0, it's a solution
        if np.all(A[r][:cols_e] == 0):
            sol = A[r][cols_e:]

            # If it's a solution, we can't have all 0s
            if np.any(sol):
                solutions.append(sol)

    return solutions


def quadratic_sieve(n: int) -> tuple[int, int, int, int]:
    """
    Fatora um número N em dois fatores utilizando o Crivo Quadrático.

    Eficiente para fatorar semiprimos.

    Retorna uma tupla (f1, f2, x, y) onde f1 e f2 são os fatores de N e x e y são os valores que satisfazem a relação x^2 = y^2 (mod N).

    Complexidade: O(e^sqrt(log n log log n)) onde e é o número de euler e n é o número a ser fatorado.
    """

    # Get approx B value
    b: int = approximate_B_value(n)

    # STEP 1: Relation building
    # Starting polynomial value
    a: int = math.ceil(math.sqrt(n))

    # Factor base
    factor_base: list[int] = build_factor_base(n, b)

    print(f"Limite superior para primos no crivo (B): {b}")
    print(f"Primos na base de fatores (existem e podem aparecer na fatoração): {len(factor_base)}")

    # Pre-compute roots for every prime in factor_base with tonneli-shanks
    factor_base_roots: dict[int, tuple[int, int]] = get_factor_base_roots(n, factor_base)

    # Start sieving
    smooth: list[tuple[int, int]] = list()
    offset: int = 0

    # Avoid infinite loop if no more smooth numbers are found
    attempts: int = 40
    old_smooth_len: int = 0

    # Sieve t^2 -n polynomials until we get enough smooth numbers
    # We will do this in batches of size b so we can save memory
    while attempts:
        # Build polynomial list to sieve
        ls, ls_divisors = build_list_to_sieve(offset, n, a, b, factor_base_roots)

        # Sieve list looking for smooth numbers
        for idx, s in enumerate(ls, offset * b):
            s_copy = s

            # No divisors
            if not ls_divisors[s]:
                continue

            # Sieve until we can't divide anymore
            for f in ls_divisors[s]:
                s = sieve(s, f)

            # If we sieved to 1, it's smooth
            if s == 1:
                smooth.append((a + idx, s_copy))

        # Check progress stall
        if len(smooth) == old_smooth_len:
            attempts -= 1
        else:
            attempts = 40  # Reset

        old_smooth_len = len(smooth)

        # We have enough smooth numbers (5 more to avoid edge cases)
        if len(smooth) >= len(factor_base) + 5:
            break

        # Not enough, try next offset
        offset += 1

    print(f"Quantidade de números testados no crivo: {offset * b + len(ls)}")

    # Relationship building complete
    # STEP 2: Elimination
    # Build exponent matrix
    exp_matrix = np.zeros((len(smooth), len(factor_base)), dtype=int)

    for i, (_, s) in enumerate(smooth):
        for j, f in enumerate(factor_base):
            while s % f == 0:  # Count factor divisions
                exp_matrix[i][j] += 1
                s //= f

    # Find solutions for w @ A = 0 in Zmod2
    solutions: list[np.ndarray] = get_matrix_kernel_solutions(exp_matrix)

    # STEP 3: GCD calculation
    for sol in solutions:
        # Convert solution to smooth number indices
        s_idx: int = [idx for idx, s in enumerate(sol) if s == 1]

        # x is the sqrt of the product of the smooth numbers that are in the solution
        x: int = reduce(
            lambda x, y: x * y,
            [smooth[i][0] for i in s_idx], 
            1
        )

        # We will calculate y using the exponents (this way we avoid sqrt, which may fail for large numbers)
        y_exp: int = reduce(
            lambda x, y: x + y,
            [exp_matrix[i] for i in s_idx],
            np.zeros(len(factor_base)),
        )

        # Dividing exponents by 2 gives us the sqrt (magic)
        y_exp //= 2

        # Calculate y
        y: int = 1

        for i, exp in enumerate(y_exp):
            y *= factor_base[i] ** int(exp)

        # For the final step, try to find factors
        f1: int = math.gcd(x - y, n)
        f2: int = math.gcd(x + y, n)

        if f1 == 1 or f2 == 1:
            continue

        # Stop after finding factors
        return (f1, f2, x, y)

    # Failure
    return (0, 0, 0, 0)


if __name__ == "__main__":
    # Load first primes (114155 first primes)
    FIRST_PRIMES = pickle.load(open("primes.pkl", "rb"))

    # Limit to first 2500 primes
    FIRST_PRIMES = FIRST_PRIMES[:2500]

    # Read semiprime from input
    sp = int(input())

    f1, f2, x, y = quadratic_sieve(sp)

    # Results
    try:
        print(f"x = {x}")
        print(f"y = {y}")
    except:
        print("x e y são muito grandes para serem exibidos")

    print(f"mdc(x - y, N) = {f1}")
    print(f"mdc(x + y, N) = {f2}")

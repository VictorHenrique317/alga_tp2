import numpy as np
from math import sqrt, log, exp, gcd
import math
from functools import reduce
from sympy import Matrix
from collections import defaultdict

import pickle

FIRST_PRIMES: list[int] = list()

def powMod(a: int, k: int, M: int) -> int:
    """
    Calculate a^k mod M in log(k) time.

    Parameters
    -------
    `a`: base.
    `k`: power.
    `M`: divisor.

    Returns
    -------
    `remainder`: result of a^k mod M.
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

def legendre_symbol(a: int, p: int):
    """
    Calcula o Símbolo de Legendre (a/p)
    """
    ls = pow(a, (p - 1) // 2, p)
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
    c = pow(z, q, p)
    t = pow(n, q, p)
    r = pow(n, (q + 1) // 2, p)
    
    while t != 1:
        # Encontra o menor i tal que t^(2^i) = 1 (mod p)
        i = 1
        temp = (t * t) % p
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
        
        # Atualiza b, t, r
        b = pow(c, 2**(m - i - 1), p)
        m = i
        c = (b * b) % p
        t = (t * c) % p
        r = (r * b) % p
    
    # First root is r, second is -r mod p
    return (r, -r % p)

def sieve(no, p):
    while no % p == 0:
        no //= p
    return no

def approximate_B_value(n: int) -> int:
    l = exp(sqrt(log(n) * log(log(n))))

    b = math.pow(l, 1 / sqrt(2))

    return math.ceil(b) + 1

def build_factor_base(n: int, b: int) -> list[int]:
    fb: list[int] = [2]

    for p in FIRST_PRIMES[1:]:
        if p > b:
            break

        # Check if p is a quadratic residue (Euler's Criteria)
        if powMod(n, (p - 1) // 2, p) != 1:
            continue

        # Add to factor base if n is a quadratic residue
        fb.append(p)

    return fb

def get_matrix_kernel_solutions(matrix: np.ndarray) -> list[np.ndarray]:
    m = Matrix(np.mod(matrix.T, 2))
    
    m_rref, base_cols = m.rref()

    m_rref = np.mod(np.array(m_rref), 2)

    rows, cols = m_rref.shape

    free_cols = [i for i in range(cols) if i not in base_cols]

    # extract base equations
    base_eq: dict[int, list[int]] = defaultdict(list)

    for row in range(rows):
        b_col = -1

        for col in range(cols):
            if m_rref[row][col] == 1 and b_col == -1:
                b_col = col
                continue
            
            if m_rref[row][col] == 1:
                base_eq[b_col].append(col)

    # find sols by iterating over free cols values
    sol = np.zeros(cols, dtype=int)

    sol_list = []

    # trivial solution, base cols are 1 if not dependent
    for col in range(cols):
        if col in base_cols and col not in base_eq:
            sol[col] = 1

    sol_list.append(sol)
    sol = np.zeros(cols, dtype=int)

    for perm in range(2 ** len(free_cols)):
        for i, col in enumerate(free_cols):
            sol[col] = (perm >> i) & 1

        for b_col, dep_cols in base_eq.items():
            sol[b_col] = 0

            for col in dep_cols:
                sol[b_col] ^= sol[col]

        if np.all(sol == 0):
            continue

        sol_list.append(sol)
        sol = np.zeros(cols, dtype=int)

    return sol_list

def quadratic_sieve(n: int):
    # Get approx B value
    b = approximate_B_value(n)

    # STEP 1: Relation building
    # We will use t^2 - n polynomial
    a: int = math.ceil(sqrt(n))

    # List to sieve # TODO: calculate on demand
    ls: list[int] = [(a + i)**2 - n for i in range(b * 5)]

    # Factor base
    fb: list[int] = build_factor_base(n, b)

    print(f"Limite superior para primos no crivo (B-smooth): {b}")
    print(f"Primos na base de fatores (existem e podem aparecer na fatoração): {len(fb)}")

    # Find roots for every fb with tonneli-shanks and convert them to indexes,
    # this way we can avoid trial divisions when sieving
    # We build a lookup table that maps (number in ls -> fb primes that will divide it)
    ls_div: dict[int, list[int]] = defaultdict(list)

    for f in fb:
        # Test 2 manually
        if f == 2:
            for s in ls:
                if s % 2 == 0:
                    ls_div[s].append(2)
            continue
        
        # Get roots
        r1, r2 = tonelli_shank(n, f)
        
        # Indexes for ls
        idx1, idx2 = (r1 - a) % f, (r2 - a) % f

        # Build lookup for ls
        while idx1 < len(ls):
            ls_div[ls[idx1]].append(f)
            idx1 += f

        while idx2 < len(ls):
            ls_div[ls[idx2]].append(f)
            idx2 += f
        
    # Sieve ls
    smooth: list[tuple[int, int]] = list()

    for i, s in enumerate(ls):
        s_copy = s

        # Not divisible by any factor base prime
        if not ls_div[s]:
            continue

        # Sieve until we can't divide anymore
        for f in ls_div[s]:
            s = sieve(s, f)

        # If we sieved to 1, it's smooth
        if s == 1:
            smooth.append((a + i, s_copy))

    # Relationship building complete
    # STEP 2: Elimination
    # Build exponent matrix
    e_matrix = np.zeros((len(smooth), len(fb)), dtype=int)

    for i, (_, s) in enumerate(smooth):
        for j, f in enumerate(fb):
            while s % f == 0:
                e_matrix[i][j] += 1
                s //= f

    # Find solutions for w @ A = 0 in Zmod2
    # Possibly more than one
    solutions: list[np.ndarray] = get_matrix_kernel_solutions(e_matrix)

    # STEP 3: GCD calculation
    for sol in solutions:
        # Convert solutions to indices
        s_idx = [i for i, s in enumerate(sol) if s == 1]
        
        # x is the sqrt of the product of the smooth numbers that are in the solution
        x: int = reduce(lambda x, y: x * y, [smooth[i][0] for i in s_idx], 1)

        # We will calculate y using the exponents (this way we avoid sqrt)
        y_exp = reduce(lambda x, y: x + y, [e_matrix[i] for i in s_idx], np.zeros(len(fb)))

        # Dividing exponents by 2 gives us the sqrt
        y_exp //= 2

        # Calculate y
        y = 1

        for i, exp in enumerate(y_exp):
            y *= fb[i] ** int(exp)

        # For the final step, try to find factors
        # Calculate gcd
        f1 = gcd(x - y, n)
        f2 = gcd(x + y, n)

        if f1 == 1 or f2 == 1:
            continue

        # Stop after finding factors
        return (f1, f2, x, y)
    
    # Failure
    return (0, 0, 0, 0)

if __name__ == "__main__":
    # Load first primes (114155 first primes)
    FIRST_PRIMES = pickle.load(open("primes.pkl", "rb"))

    # Limit to first 1000 primes
    FIRST_PRIMES = FIRST_PRIMES[:1000]

    # Read semiprime from imput
    sp = int(input())

    f1, f2, x, y = quadratic_sieve(sp)

    print(f"mdc(x - y, N) = {f1}")
    print(f"mdc(x + y, N) = {f2}")
    print(f"x = {x}, y = {y}")

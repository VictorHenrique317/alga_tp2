import numpy as np

class LinearEquation:
    """
    Representa uma equação linear da forma x0_coef * x0 + x1_coef * x1 = y.
    
    Atributos:
        x0_coef (int): Coeficiente de x0 na equação.
        x1_coef (int): Coeficiente de x1 na equação.
        y (int): Valor constante da equação.
    """

    def __init__(self, x0_coef: int, x1_coef:int , y:int):
        self.x0_coef = x0_coef
        self.x1_coef = x1_coef
        self.y = y

    def __str__(self):
            """
            Retorna uma representação em string do objeto.

            A representação em string segue o formato:
            "{coeficiente_x0}x0 + {coeficiente_x1}x1 = {valor_y}"

            Returns:
                str: A representação em string do objeto.
            """
            return f"{self.x0_coef}x0 + {self.x1_coef}x1 = {self.y}"

class LinearEquationSystem:
    def __init__(self, equations: list[LinearEquation]):
        """
        Inicializa um sistema de equações lineares.

        Args:
            equations (list[LinearEquation]): Uma lista de equações lineares.
        """
        self.equations = equations

    def __str__(self):
        """
        Retorna uma representação em string do sistema de equações lineares.

        Returns:
            str: A representação em string do sistema de equações lineares.
        """
        return "\n".join(str(eq) for eq in self.equations)

    def solve(self) -> np.ndarray:
        """
        Resolve o sistema de equações lineares.

        Returns:
            np.ndarray: Um array numpy contendo as soluções do sistema de equações lineares.
        """
        A = np.array([[eq.x0_coef, eq.x1_coef] for eq in self.equations])
        B = np.array([eq.y for eq in self.equations])
        return np.linalg.solve(A, B)


def legendre_symbol(a, p):
    """Calcula o Símbolo de Legendre (a/p)
    """
    ls = pow(a, (p - 1) // 2, p)
    return -1 if ls == p - 1 else ls

def tonelli_shank(n, p):
    """Encontra uma raiz quadrada de n módulo p, se existir.
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
    
    return r


equation1 = LinearEquation(1, 1, 35)
equation2 = LinearEquation(2, 4, 94)
system = LinearEquationSystem([equation1, equation2])
print(system.solve())
print(tonelli_shank(13 ,179))

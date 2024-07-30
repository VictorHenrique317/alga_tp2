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

equation1 = LinearEquation(1, 1, 35)
equation2 = LinearEquation(2, 4, 94)
system = LinearEquationSystem([equation1, equation2])
print(system.solve())

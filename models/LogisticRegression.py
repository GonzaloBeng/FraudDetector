from matplotlib import pyplot as plt
import numpy as np
import time

from sklearn.base import accuracy_score

class LogisticRegression:

    def __init__(
        self, step_size=0.2, max_iter=100, eps=1e-5, theta_0=None, verbose=True, method="newton"
    ):
        """
        Args:
            step_size: tamaño del paso para cada iteración.
            max_iter: cantidad máxima de iteraciones.
            eps: umbral para convergencia.
            theta_0: valor inicial de tita. Si None, use el vector cero.
            verbose: imprimir valores de pérdida durante el entrenamiento.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.method = method
        self.costo = []
        self.coeficientes = []
        self.contador_iteraciones = 0
        self.accuracy = []


    def fit(self, x, y, alpha=1):
        """Corre el método de Newton para minimizar J(tita) para reg log.

        Args:
            x: ejemplos de entrenamiento (features solamente). Tamaño (m, n).
            y: etiquetas de ejemplos de entrenamiento. Tamaño (m,).
        """

        m, n = x.shape # dimensiones de x

        if self.theta is None: # si theta es None, inicializa theta en vector de ceros
            self.theta = np.zeros(n)
            # self.theta = np.random.random(n) da error al calcular la inversa de la matriz hessiana

        sigmoide = lambda theta: 1 / (1 + np.exp(-theta @ x.T)) # funcion sigmoide

        costo = lambda sigm_vals: -1 / m * (y @ np.log(sigm_vals) + (1 - y) @ np.log(1 - sigm_vals)) # funcion de costo

        gradiente = lambda sigm_vals: 1 / m * (sigm_vals - y) @ x # gradiente

        hessiano = lambda sigm_vals: 1 / m * (sigm_vals * (1 - sigm_vals) * x.T @ x) # hessiano

        error = 1 # inicializa el error en 1

        inicio = time.time() # guarda el tiempo de inicio
        if self.method == "newton":
            # comienza con las iteraciones con el metodo de Newton
            while error > self.eps and self.contador_iteraciones < self.max_iter: # iterar hasta que el error sea menor a epsilon o se llegue al maximo de iteraciones
                pred = sigmoide(self.theta) # calculo de la sigmoide
                grad = gradiente(sigm_vals=pred) # calculo del gradiente
                hess = hessiano(sigm_vals=pred) # calculo del hessiano
                hess_inv = np.linalg.inv(hess) # inversa del hessiano
                new_theta = self.theta - hess_inv @ grad # calculo de theta nuevo
                error = np.linalg.norm(new_theta - self.theta) # calculo del error
                self.theta = new_theta # actualizacion de theta
                self.contador_iteraciones += 1 # actualizacion de contador de iteraciones

                new_pred = sigmoide(self.theta) # calculo de la sigmoide
                # guarda los valores de theta y el costo
                self.coeficientes.append(self.theta) # guarda los valores de theta
                self.costo.append(costo(sigm_vals=new_pred)) # guarda el costo

                # calcula el accuracy en cada iteracion. Corte 0.5
                new_pred = new_pred / alpha # aplica el factor de correccion
                new_pred[new_pred >= 0.5] = 1
                new_pred[new_pred < 0.5] = 0
                self.accuracy.append(accuracy_score(y, new_pred)) # guarda el accuracy
                
        elif self.method == "gradiente":
            for _ in range(self.max_iter):
                pred = sigmoide(self.theta) # calculo de la sigmoide
                grad = gradiente(sigm_vals=pred) # calculo del gradiente
                new_theta = self.theta - self.step_size * grad # calculo de theta nuevo
                error = np.linalg.norm(new_theta - self.theta) # calculo del error
                self.theta = new_theta # actualizacion de theta
                self.contador_iteraciones += 1 # actualizacion de contador de iteraciones

                new_pred = sigmoide(self.theta) # calculo de la sigmoide
                # guarda los valores de theta y el costo
                self.coeficientes.append(self.theta) # guarda los valores de theta
                self.costo.append(costo(sigm_vals=new_pred)) # guarda el costo

                # calcula el accuracy en cada iteracion. Corte 0.5
                new_pred = new_pred / alpha # aplica el factor de correccion
                new_pred[new_pred >= 0.5] = 1
                new_pred[new_pred < 0.5] = 0
                self.accuracy.append(accuracy_score(y, new_pred)) # guarda el accuracy

                if error < self.eps:
                    break
        
        final = time.time() # guarda el tiempo final
        delta = final - inicio # calcula el tiempo transcurrido

        if self.verbose: # si verbose es True, imprime los resultados
            print("Terminó en", self.contador_iteraciones, "iteraciones")
            print("Error:", error)
            print("Theta:", self.theta)
            print("Tiempo transcurrido:", delta)


    def predict(self, x):
        """Hace una predicción sobre x nuevos.

        Args:
            x: entradas de tamaño (m, n).

        Returns:
            Salidas de tamaño (m,).
        """

        prob_1 = lambda theta: 1 / (1 + np.exp(-theta @ x.T)) # funcion que calcula la probabilidad de que sea 1

        probs = prob_1(self.theta) # calcula la probabilidad de que sea 1
        return probs


    def graficos(self, pred_path):
        """Crea los siguientes gráficos.

        - Costo vs Iteraciones
        - Accuracy de entrenamiento vs Iteraciones
        - Evolución features (sin graficar el intercept)
        Args:
            pred_path: directorio para guardar las imágenes.
        """

        # Costo vs Iteraciones
        plt.clf()  # limpia el gráfico anterior
        plt.plot(self.costo)
        plt.xlabel("Iteraciones")
        plt.ylabel("Costo")
        plt.title("Costo vs Iteraciones")
        plt.savefig(pred_path + "/costo_iteraciones.png")

        # Accuracy de entrenamiento vs Iteraciones
        plt.clf()  # limpia el gráfico anterior
        plt.plot(self.accuracy)
        plt.xlabel("Iteraciones")
        plt.ylabel("Accuracy")
        plt.title("Accuracy de entrenamiento vs Iteraciones")
        plt.legend(["Accuracy con corte en 0.5"])
        plt.savefig(pred_path + "/accuracy_iteraciones.png")

        # Evolución features (sin graficar el intercept)
        plt.clf()  # limpia el gráfico anterior
        for i in range(1, len(self.coeficientes[0])):
            plt.plot([theta[i] for theta in self.coeficientes])
        plt.xlabel("Iteraciones")
        plt.ylabel("Coeficientes")
        plt.title("Evolución features")
        plt.legend(["Feature 1", "Feature 2"])
        plt.savefig(pred_path + "/evolucion_features.png")

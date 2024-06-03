import numpy as np
import matplotlib.pyplot as plt

class RedHopfield:
    def __init__(self, tamano):
        self.tamano = tamano
        self.pesos = np.zeros((tamano, tamano))
    
    def entrenar(self, patrones):
        num_patrones = len(patrones)
        for patron in patrones:
            patron_bipolar = self._binario_a_bipolar(patron)
            self.pesos += np.outer(patron_bipolar, patron_bipolar)
        self.pesos /= num_patrones
        np.fill_diagonal(self.pesos, 0)
    
    def predecir(self, patron, pasos=10):
        patron_bipolar = self._binario_a_bipolar(patron)
        for _ in range(pasos):
            patron_actualizado = np.sign(self.pesos @ patron_bipolar)
            patron_actualizado[patron_actualizado == 0] = 1
            if np.array_equal(patron_actualizado, patron_bipolar):
                break
            patron_bipolar = patron_actualizado
        return self._bipolar_a_binario(patron_bipolar)
    
    def _binario_a_bipolar(self, patron_binario):
        return np.where(np.array(list(patron_binario), dtype=int) == 0, -1, 1)

    def _bipolar_a_binario(self, patron_bipolar):
        return ''.join(['1' if x == 1 else '0' for x in patron_bipolar])

def mostrar_patron(patron, titulo, tamano=10):
    matriz_patron = np.array(list(patron), dtype=int).reshape((tamano, tamano))
    plt.imshow(matriz_patron, cmap='binary')
    plt.title(titulo)
    plt.axis('off')
    plt.show()

def mutar_patron(patron, tasa_mutacion):
    lista_patron = list(patron)
    num_mutaciones = int(len(patron) * tasa_mutacion)
    indices_a_mutar = np.random.choice(len(patron), num_mutaciones, replace=False)
    for indice in indices_a_mutar:
        lista_patron[indice] = '1' if lista_patron[indice] == '0' else '0'
    return ''.join(lista_patron)

def main():
    # Definición de patrones (caracteres binarios 10x10)
    patrones = [
        '0110100010' * 10,
        '1001001100' * 10,
        '1110101110' * 10
    ]

    # Inicializar y entrenar la red
    hopfield = RedHopfield(tamano=100)
    hopfield.entrenar(patrones)

    for patron in patrones:
        mutado = mutar_patron(patron, 0.3)
        recuperado = hopfield.predecir(mutado)
        
        print("\nPatrón Original:")
        mostrar_patron(patron, "Patrón Original")
        print("\nPatrón Mutado:")
        mostrar_patron(mutado, "Patrón Mutado")
        print("\nPatrón Recuperado:")
        mostrar_patron(recuperado, "Patrón Recuperado")

if __name__ == "__main__":
    main()

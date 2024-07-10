import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Variables de Entrada
fiebre = ctrl.Antecedent(np.arange(35, 42, 0.1), 'fiebre')
dor_de_cabeca = ctrl.Antecedent(np.arange(0, 11, 1), 'dorlor de cabeza')
tos = ctrl.Antecedent(np.arange(0, 11, 1), 'tos')

# Variable de Salida
diagnostico = ctrl.Consequent(np.arange(0, 101, 1), 'diagnostico')

# Define las funciones de pertenencia para la fiebre
fiebre['normal'] = fuzz.trimf(fiebre.universe, [35, 36.5, 37.5])
fiebre['moderada'] = fuzz.trimf(fiebre.universe, [36, 37.5, 38.5])
fiebre['alta'] = fuzz.trimf(fiebre.universe, [38, 39.5, 41])

# Define las funciones de pertenencia para el dolor de cabeza
dor_de_cabeca['leve'] = fuzz.trimf(dor_de_cabeca.universe, [0, 2, 4])
dor_de_cabeca['moderada'] = fuzz.trimf(dor_de_cabeca.universe, [3, 5, 7])
dor_de_cabeca['intensa'] = fuzz.trimf(dor_de_cabeca.universe, [6, 8, 10])

# Define las funciones de pertenencia para la tosse
tos['ocasional'] = fuzz.trimf(tos.universe, [0, 2, 4])
tos['frecuente'] = fuzz.trimf(tos.universe, [3, 5, 7])
tos['persistente'] = fuzz.trimf(tos.universe, [6, 8, 10])

# Define las funciones de pertenencia para el diagnóstico
diagnostico['bajo'] = fuzz.trimf(diagnostico.universe, [0, 0, 50])
diagnostico['medio'] = fuzz.trimf(diagnostico.universe, [25, 50, 75])
diagnostico['alto'] = fuzz.trimf(diagnostico.universe, [50, 100, 100])

# Reglas difusas
rule1 = ctrl.Rule(fiebre['alta'] | dor_de_cabeca['intensa'] | tos['persistente'], diagnostico['alto'])
rule2 = ctrl.Rule(fiebre['moderada'] | dor_de_cabeca['moderada'] | tos['frecuente'], diagnostico['medio'])
rule3 = ctrl.Rule(fiebre['normal'] | dor_de_cabeca['leve'] | tos['ocasional'], diagnostico['bajo'])

# Crea el sistema de control difuso
diagnostico_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
diagnostico_simulador = ctrl.ControlSystemSimulation(diagnostico_ctrl)

# Ingresa los valores de entrada
diagnostico_simulador.input['fiebre'] = 39.2
diagnostico_simulador.input['dorlor de cabeza'] = 6
diagnostico_simulador.input['tos'] = 2

# Computa el resultado
diagnostico_simulador.compute()

# Muestra las funciones de pertenencia de las variables de entrada
fiebre.view()
dor_de_cabeca.view()
tos.view()

# Muestra las funciones de pertenencia de la variable de salida (diagnóstico)
diagnostico.view()

# Muestra la salida del diagnóstico con la simulación
print("Diagnóstico:", diagnostico_simulador.output['diagnostico'])
diagnostico.view(sim=diagnostico_simulador)

# Mostrar todos los gráficos al mismo tiempo
plt.show()

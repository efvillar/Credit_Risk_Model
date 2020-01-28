# MODELO DE RIESGO DE CRÉDITO PARA PERSONA NATURAL
Modelo de Riesgo de Crédito para Personas Naturales
Este repositorio contiene los elementos teoricos y resultados de un modelo de riesgo para créditos de personas naturales.
Los datos del modelo fueron obtenidos de https://www.kaggle.com/wendykan/lending-club-loan-data.

## VARIABLES DE ENTRADA

Las variables de entrada al modelo son:

1. Calificación Créditicia a Priori del deudor obtenida de un Buró de Crédito
2. Datos Demográficos del deudor
3. Datos transaccionales del deudor

## VARIABLES DE SALIDA

**1. Probability of Default (PD):**  
Se empleará un modelo de regresión logistica para la estimación de la probabilidad de incumplimiento de un deudor, dados los datos de entrada al modelo.

**2. Loss Given Default (LGD):**
Modelo combinado en donde inicialmente se estimará un modelo para determinar si hay probabilidad alta de default, en cuyo caso se corre el segundo modelo, menos sesgado, donde se estimará el porcentaje de pérdida en caso de default, sobre el capital a la fecha.

**3. Exposure at Default (EAD):**
Este modelo estimará el saldo de la deuda en el momento del default.

**4.  Pérdida Esperada por Crédito (PEC):**
Será el producto PE = LGD * EAD

**5. Pérdida esperada Portafolio de Créditos (PEPC):**
Es la sumatoria de las pérdidas esperadas de todos los clientes, asumiendo independencia.  Posteriormente con mayor cantidad de datos se podría eliminar el supuesto de independencia.
\sum\PEC

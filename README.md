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

**6. Assesing Population Stability (APS):**
Indicador para realizar monitoreo de la necesidad de reentrenar los modelos de crédito, debido al cambio en la distribución de probabilidad de los datos de entrada

**7. Modelo de Credit Allocation:**
Se usará el criterio de máximización de riqueza pero de modo fraccional.  Fractional Kelly Criterion.

**8. Modelo de Pricing del Crédito:**
El precio del crédito debe ser una función de la perdida esperada y el nivel de precios del dinero en la economía


## DEPLOYMENT

La implementación será en Python y se realizará el deployment via un webservice

El entrenamiento de los modelos se realziará cada N meses de acuerdo con el indicador APS.   Los modelos a entrenar se exportarán a un servidor para que consuman los datos que sean enviador por la entidad que requiere los cálculos.  Se deberá establecer un protocolo de intercambio de datos para recibir los inputs del modelo y entrear los outputs.

Inicialmente de las 8 medidas el servicio calculará las numero 1, 2, 3, 4, 6, 7, y 8.  La número 5 (PEPC) se puede calcular directamente por parte de la entidad prestamista.

Este diseño modular permite la actualización del modelo de riesgo sin afectar la operación del prestamista.

## DIAGRAMA SIMPLIFICADO DE LA OPERACIÓN

![Diagrama Simplificado Operacion](/Esquema.PNG)




# Ecuaciones Diferenciales Parciales II (EDP II)

Repositorio dedicado al estudio, implementación y simulación de
**Ecuaciones Diferenciales Parciales (EDP)**, incluyendo desarrollo teórico y
práctico, resolución analítica y métodos numéricos.

## Objetivo

Este repositorio tiene como propósito:

- Documentar tareas y ejercicios de la materia de _EDP II_
- Implementar soluciones analíticas de problemas clásicos
- Desarrollar simulaciones numéricas Python
- Visualizar fenómenos físicos modelados por EDP

---

## Contenido

El proyecto está estructurado de la siguiente manera:

```
EDP-II
├───eqsolver
│   ├───core
│   ├───methods
│   │   └───adomian
│   ├───tests
│   └───utils
|
├───reports
│   ├───tarea-1
│   │   └───figures
│   └───tarea-2
│       └───figures
|
├───scripts
|
└───visualization
    ├───one_dimensional_time_eqs
    ├───two_dimensional_time_eqs
    └───utils
```

## Instalación y Configuración

### Prerrequisitos
```bash
# Clonar el repositorio
git clone https://github.com/HazelS002/EDP-II.git
cd EDP-II

# Crear entorno de Conda
conda env create -f environment.yml

# Activar entorno de Conda
conda activate math
```

### Dependencias
- **NumPy**: Cálculos numéricos y operaciones con arrays de imágenes
- **Matplotlib**: Visualización
- **scipy**: Funciones matemáticas avanzadas

## Ejecución
### Generar PDF (requiere pdflatex)

Para compilar el archivo _tex_ a _pdf_ de la tarea $n$:
```bash
cd "reports/tarea-<n>/"
pdflatex main.tex
```

### Ejecutar scripts de Python
Para ejecutar el script de la tarea $n$:
```bash
python -m scripts.tarea-<n>    # (requiere pdflatex)
```
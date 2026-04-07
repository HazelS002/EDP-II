# 📘 Ecuaciones Diferenciales Parciales II (EDP II)

Repositorio dedicado al estudio, implementación y simulación de
**Ecuaciones Diferenciales Parciales (EDP)**, incluyendo desarrollo teórico y
práctico, resolución analítica y métodos numéricos.

## 📌 Objetivo

Este repositorio tiene como propósito:

- Documentar tareas y ejercicios de la materia de _EDP II_
- Implementar soluciones analíticas de problemas clásicos
- Desarrollar simulaciones numéricas Python
- Visualizar fenómenos físicos modelados por EDP

---

## 🧠 Contenido

El proyecto está estructurado de la siguiente manera:

```
EDP-II
├───reportes-tareas
│   ├───tarea-1
│   │   └───figures
│   └───tarea-2
└───scripts
    ├───tarea-1
    │   └───figures
    └───tarea-2
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
cd "reportes-tareas/tarea-<n>/"
pdflatex main.tex
```

### Ejecutar scripts de Python
Para ejecutar el script de la tarea $n$:
```bash
cd "scripts/tarea-<n>/"
python tarea-<n>.py    # (requiere pdflatex)
```
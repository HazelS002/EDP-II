from dolfin import TimeSeries, Function

def get_timeserie(space, dirname):
    ts = TimeSeries(dirname)    # Abrir los TimeSeries en modo lectura
    times = ts.vector_times()   # obtener tiempos
    u = []                      # para guardar los valores de cada tiempo

    for t in times:
        u_t = Function(space)   # Crear funciones temporales para cada tiempo
        ts.retrieve(u_t.vector(), t) # Recuperar vectores en el instante t
        u.append(u_t)

    return times, u
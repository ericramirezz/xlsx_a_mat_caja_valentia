"""
convertir_xlsx_a_mat.py

Convierte tablas de Excel (.xlsx) al formato .mat que espera la Caja de Valentia
(leer_mat_a_df), compatible con MATLAB R2011b.

Formato de salida (un .mat por cada Excel):
    - UNA sola variable 2D (N filas x 8 columnas, double), en este orden:
      Ensayo | Lado | Estim Electrico | Latencia | Tiempo Absoluto |
      Palancas Izq | Palancas Der | Desplazamiento
    - Formato MAT-file v5 (el que MATLAB 2011b lee sin problema).
    - El .mat conserva el nombre del Excel (Rata01.xlsx -> Rata01.mat).

Uso con ventanas (explorador de archivos):
    python convertir_xlsx_a_mat.py

Uso por linea de comandos:
    python convertir_xlsx_a_mat.py archivo.xlsx
    python convertir_xlsx_a_mat.py carpeta_con_excels -o carpeta_salida
    python convertir_xlsx_a_mat.py archivo.xlsx --hoja Hoja1 --variable datos
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.io

# Nombres esperados por leer_mat_a_df (solo informativos: la lectura es por posicion)
COLUMNAS_ESPERADAS = [
    'Ensayo', 'Lado', 'Estim Electrico', 'Latencia',
    'Tiempo Absoluto', 'Palancas Izq', 'Palancas Der', 'Desplazamiento'
]
N_COLUMNAS = len(COLUMNAS_ESPERADAS)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def listar_xlsx(carpeta):
    """Devuelve los .xlsx de una carpeta, ordenados (ignora temporales '~$' de Excel)."""
    return sorted(
        os.path.join(carpeta, f) for f in os.listdir(carpeta)
        if f.lower().endswith('.xlsx') and not f.startswith('~$')
    )


def _elegir_hoja(ruta_xlsx, hoja=None):
    """Devuelve el DataFrame (sin encabezado interpretado) de la hoja a convertir.
    Si no se indica hoja, usa la primera que tenga datos (ignora hojas vacias)."""
    xls = pd.ExcelFile(ruta_xlsx, engine='openpyxl')

    if hoja is not None:
        if hoja not in xls.sheet_names:
            raise ValueError(f"La hoja '{hoja}' no existe. Hojas disponibles: {xls.sheet_names}")
        return xls.parse(hoja, header=None), hoja

    for nombre in xls.sheet_names:
        df = xls.parse(nombre, header=None).dropna(how='all')
        if not df.empty:
            return xls.parse(nombre, header=None), nombre

    raise ValueError("Ninguna hoja del archivo contiene datos.")


def excel_a_matriz(ruta_xlsx, hoja=None, log_fn=print):
    """Lee el Excel y devuelve (matriz float64 de N x 8, nombre de hoja usada)."""
    df, nombre_hoja = _elegir_hoja(ruta_xlsx, hoja)

    # Quitamos filas y columnas completamente vacias (tipico de Excel viejos)
    df = df.dropna(how='all').dropna(axis=1, how='all')

    # Si la primera fila es encabezado (texto), la descartamos
    primera_fila = pd.to_numeric(df.iloc[0], errors='coerce')
    if primera_fila.isna().any():
        df = df.iloc[1:]

    if df.shape[1] != N_COLUMNAS:
        raise ValueError(
            f"La hoja '{nombre_hoja}' tiene {df.shape[1]} columnas con datos, "
            f"pero se esperan {N_COLUMNAS}: {COLUMNAS_ESPERADAS}"
        )

    # Convertimos todo a numerico; si hay texto en las celdas de datos, avisamos donde
    numerico = df.apply(pd.to_numeric, errors='coerce')
    invalidas = numerico.isna() & df.notna()
    if invalidas.any().any():
        fila, col = np.argwhere(invalidas.to_numpy())[0]
        raise ValueError(
            f"Valor no numerico en hoja '{nombre_hoja}', fila de Excel "
            f"{df.index[fila] + 1}, columna {col + 1}: {df.iloc[fila, col]!r}"
        )

    matriz = numerico.to_numpy(dtype=np.float64)  # celdas vacias -> NaN (MATLAB las lee como NaN)
    if np.isnan(matriz).any():
        log_fn(f"  Aviso: {int(np.isnan(matriz).sum())} celda(s) vacia(s) se guardaran como NaN.")

    return matriz, nombre_hoja


def guardar_mat(matriz, ruta_mat, variable='datos'):
    """Guarda la matriz como MAT-file v5 con una unica variable."""
    scipy.io.savemat(ruta_mat, {variable: matriz}, format='5', oned_as='column')


def verificar_mat(ruta_mat, matriz_original):
    """Relee el .mat igual que lo hace leer_mat_a_df y compara con el original."""
    mat_data = scipy.io.loadmat(ruta_mat)
    datos = next((v for k, v in mat_data.items() if not k.startswith('__')), None)
    if datos is None:
        raise RuntimeError("El .mat generado no contiene variables.")
    if datos.shape != matriz_original.shape:
        raise RuntimeError(f"Dimensiones distintas: {datos.shape} vs {matriz_original.shape}")
    if not np.allclose(datos, matriz_original, equal_nan=True):
        raise RuntimeError("Los valores del .mat no coinciden con el Excel.")
    return datos.shape


def convertir_archivo(ruta_xlsx, carpeta_salida=None, hoja=None, variable='datos', log_fn=print):
    """Convierte un .xlsx a .mat. Devuelve la ruta del .mat o None si falla."""
    archivo = os.path.basename(ruta_xlsx)
    nombre = os.path.splitext(archivo)[0]
    carpeta_salida = carpeta_salida or os.path.dirname(os.path.abspath(ruta_xlsx))
    os.makedirs(carpeta_salida, exist_ok=True)
    ruta_mat = os.path.join(carpeta_salida, nombre + '.mat')

    log_fn(archivo)
    try:
        matriz, hoja_usada = excel_a_matriz(ruta_xlsx, hoja, log_fn)
        guardar_mat(matriz, ruta_mat, variable)
        forma = verificar_mat(ruta_mat, matriz)
        log_fn(f"  OK  hoja '{hoja_usada}' -> {os.path.basename(ruta_mat)}  "
               f"{forma[0]} filas x {forma[1]} columnas")
        return ruta_mat
    except Exception as e:
        log_fn(f"  ERROR en {archivo}: {e}")
        return None


def convertir_lote(archivos, carpeta_salida=None, hoja=None, variable='datos', log_fn=print):
    """Convierte varios .xlsx. Devuelve (cantidad_ok, lista_de_archivos_fallidos)."""
    fallidos = []
    for a in archivos:
        if convertir_archivo(a, carpeta_salida, hoja, variable, log_fn) is None:
            fallidos.append(os.path.basename(a))
    return len(archivos) - len(fallidos), fallidos


# ---------------------------------------------------------------------------
# Interfaz con ventanas (explorador de archivos)
# ---------------------------------------------------------------------------

def modo_ventanas(variable='datos', hoja=None):
    """Pide con el explorador de archivos: 1) los Excel o una carpeta, 2) donde guardar los .mat."""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("Convertir Excel a .mat")
    root.resizable(False, False)

    seleccion = {'archivos': [], 'carpeta_origen': None}
    terminado = tk.StringVar(value='')

    def elegir_archivos():
        rutas = filedialog.askopenfilenames(
            parent=root, title="Selecciona uno o varios archivos Excel",
            filetypes=[("Excel (*.xlsx)", "*.xlsx"), ("Todos los archivos", "*.*")])
        if rutas:
            seleccion['archivos'] = list(rutas)
            seleccion['carpeta_origen'] = os.path.dirname(rutas[0])
            terminado.set('ok')

    def elegir_carpeta():
        carpeta = filedialog.askdirectory(parent=root, title="Selecciona la carpeta con los Excel")
        if not carpeta:
            return
        archivos = listar_xlsx(carpeta)
        if not archivos:
            messagebox.showwarning("Sin archivos", "Esa carpeta no tiene archivos .xlsx.", parent=root)
            return
        seleccion['archivos'] = archivos
        seleccion['carpeta_origen'] = carpeta
        terminado.set('ok')

    tk.Label(root, text="¿Qué quieres convertir?", font=("Segoe UI", 11, "bold")).pack(padx=30, pady=(20, 10))
    tk.Button(root, text="Elegir archivo(s) Excel...", width=32, command=elegir_archivos).pack(padx=30, pady=4)
    tk.Button(root, text="Elegir una carpeta con Excel...", width=32, command=elegir_carpeta).pack(padx=30, pady=(4, 20))
    root.protocol("WM_DELETE_WINDOW", lambda: terminado.set('cancelar'))

    root.wait_variable(terminado)
    if terminado.get() != 'ok':
        root.destroy()
        return

    root.withdraw()
    carpeta_salida = filedialog.askdirectory(
        parent=root, title="Elige donde guardar los archivos .mat",
        initialdir=seleccion['carpeta_origen'])
    if not carpeta_salida:
        root.destroy()
        return

    log = []
    ok, fallidos = convertir_lote(seleccion['archivos'], carpeta_salida, hoja, variable, log.append)
    total = len(seleccion['archivos'])

    if fallidos:
        errores = [l.strip() for l in log if 'ERROR' in l]
        detalle = "\n".join(errores[:8]) + ("\n..." if len(errores) > 8 else "")
        messagebox.showwarning(
            "Conversión terminada con errores",
            f"Convertidos: {ok}/{total}\nGuardados en:\n{carpeta_salida}\n\n{detalle}", parent=root)
    else:
        messagebox.showinfo(
            "Conversión terminada",
            f"Convertidos: {ok}/{total}\nGuardados en:\n{carpeta_salida}", parent=root)
    root.destroy()


# ---------------------------------------------------------------------------
# Linea de comandos
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Convierte Excel (.xlsx) a .mat para la Caja de Valentia. "
                    "Sin argumentos abre ventanas para elegir los archivos.")
    p.add_argument('entrada', nargs='?', help="Archivo .xlsx o carpeta con varios .xlsx")
    p.add_argument('-o', '--salida', help="Carpeta de salida (por defecto, junto al Excel)")
    p.add_argument('--hoja', help="Nombre de la hoja a convertir (por defecto, la primera con datos)")
    p.add_argument('--variable', default='datos', help="Nombre de la variable dentro del .mat (default: datos)")
    args = p.parse_args()

    if args.entrada is None:
        modo_ventanas(args.variable, args.hoja)
        return

    if os.path.isdir(args.entrada):
        archivos = listar_xlsx(args.entrada)
        if not archivos:
            sys.exit("No se encontraron archivos .xlsx en la carpeta.")
    elif os.path.isfile(args.entrada):
        archivos = [args.entrada]
    else:
        sys.exit(f"No existe: {args.entrada}")

    ok, _ = convertir_lote(archivos, args.salida, args.hoja, args.variable)
    print(f"\nConvertidos: {ok}/{len(archivos)}")
    sys.exit(0 if ok == len(archivos) else 1)


if __name__ == '__main__':
    main()

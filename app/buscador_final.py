import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from openai import OpenAI

from modules.utils_final import procesar_orden


texto = """
Quiero ordenar la Caja Tech y Cafe.
Entrega urgente mañana en la Zona Rosa en la Alcaldía Polanco. Codigo postal es 98765
Notas: dejar en caseta de vigilancia. Dedicatoria: Feliz Cumpleaños
"""

print(procesar_orden(texto, debug=True))

import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from openai import OpenAI

from modules.utils import responder_consulta_usuario


consulta_1 = "Busco un regalo corporativo elegante con vino tinto alrededor de 80 dólares."
print("Consulta 1:")
print(consulta_1)
print()

respuesta_1 = responder_consulta_usuario(consulta_1, k=5)
print(respuesta_1)

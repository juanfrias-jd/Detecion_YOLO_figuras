"""
Descarga el dataset de Roboflow Universe (formato YOLOv8) en la carpeta data_roboflow.

Antes de ejecutar, define tu API key (NO la escribas en el código ni la subas a GitHub):
    PowerShell:   $env:ROBOFLOW_API_KEY = "tu_api_key"
Uso:
    python download_roboflow.py
"""
import os

from roboflow import Roboflow

API_KEY = os.environ.get("ROBOFLOW_API_KEY")
WORKSPACE = "aerovant"
PROJECT = "object-detection-q5uab"
VERSION = 1  # cambia al número de versión más reciente que veas en la página del dataset

if not API_KEY:
    raise SystemExit('Falta la API key. En PowerShell ejecuta primero:  $env:ROBOFLOW_API_KEY = "tu_api_key"')

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
dataset = project.version(VERSION).download("yolov8", location="data_roboflow")
print("Descargado en:", dataset.location)

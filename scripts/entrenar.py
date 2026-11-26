"""Ejecutar desde la raíz: python scripts/entrenar.py --profile complete."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación reproducible según la guía del hackathon.")
    parser.add_argument("--profile", choices=["quick", "complete"], default="complete")
    args = parser.parse_args()
    # Un ensayo rápido jamás reemplaza los resultados completos ni submission_Grupo_1.csv.
    output = None if args.profile == "complete" else Path("outputs/quick")
    run_experiment(args.profile, output_dir=output)

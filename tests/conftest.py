"""Agent-S3 test collection — barreira de regressão robusta.

Torna `pytest tests/` sempre colecionável, independentemente do ambiente:

- Ignora test_grounding_computer_use.py quando pytesseract ou o binário
  tesseract não estão disponíveis (import top-level em grounding.py quebra
  a coleção sem isso). Localmente (Beto, venv) pytesseract+tesseract estão
  instalados → os 6 testes rodam. Em CI/ambiente sem tesseract-ocr → pulam
  sem explodir a coleção.
"""
import importlib.util
import shutil


def _pytesseract_ok():
    if importlib.util.find_spec("pytesseract") is None:
        return False
    return shutil.which("tesseract") is not None


# collect_ignore (lista canônica do pytest) — caminhos relativos a este
# conftest que NÃO devem ser coletados. Definido só quando o grounding
# não puder importar, preservando os 35 testes core.
if not _pytesseract_ok():
    collect_ignore = ["test_grounding_computer_use.py"]
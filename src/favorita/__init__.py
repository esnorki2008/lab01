"""Producto de pronóstico de ventas y pedido sugerido (Corporación Favorita)."""
import sys

__version__ = "1.0.0"

if sys.platform == "win32":  # pragma: no cover
    # En Windows, cargar pyarrow antes que LightGBM provoca un conflicto de DLLs
    # (access violation al construir el Dataset). Forzamos el orden de carga.
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        pass

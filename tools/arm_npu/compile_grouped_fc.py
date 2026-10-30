"""Compile geometry-only original-group programs; never requantize a model."""
from compile_dynamic_fc import main

if __name__=='__main__':raise SystemExit(main(original_groups=True))

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.interface.app import MyApp

if __name__ == "__main__":
    MyApp().run()

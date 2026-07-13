"This module contains the class which is used to plot the results of"
"numerical simulations."

from zipfile import Path


class Plotter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

- 👋 Hi, I’m @rayenhafsi
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...

<!---
rayenhafsi/rayenhafsi is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
 contributor
@@ -13,10 +13,8 @@ jobs:
        python-version: [3.9]
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v2
        uses: actions/setup-python@v3
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
  4  
.github/workflows/release.yml
@@ -35,7 +35,7 @@ jobs:
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python 3.9
        uses: actions/setup-python@v2
        uses: actions/setup-python@v3
        with:
          python-version: 3.9
          cache: pip
@@ -61,7 +61,7 @@ jobs:
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v2
        uses: actions/setup-python@v3
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
  2  
.github/workflows/test-linux.yml
@@ -69,7 +69,7 @@ jobs:
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python 3.9
        uses: actions/setup-python@v2
        uses: actions/setup-python@v3
        with:
          python-version: 3.9
      - name: Install Coverage
  2  
.github/workflows/test-mac.yml
@@ -30,7 +30,7 @@ jobs:
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v2
        uses: actions/setup-python@v3
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
  2  
.github/workflows/test-win.yml
@@ -30,7 +30,7 @@ jobs:
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v2
        uses: actions/setup-python@v3
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
  54  
.pre-commit-config.yaml
@@ -8,15 +8,23 @@ ci:
  skip: [flake8, mypy]

repos:
  - repo: https://github.com/psf/black
    rev: 22.1.0

  - repo: https://github.com/myint/autoflake
    rev: v1.4
    hooks:
      - id: black
      - id: autoflake
        args:
          - --in-place
          - --remove-unused-variables
          - --remove-all-unused-imports
          - --expand-star-imports
          - --ignore-init-module-imports

  - repo: https://github.com/PyCQA/isort
    rev: 5.10.1
  - repo: https://github.com/asottile/pyupgrade
    rev: v2.31.0
    hooks:
      - id: isort
      - id: pyupgrade
        args: [--py38-plus]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.1.0
@@ -26,35 +34,23 @@ repos:
      - id: end-of-file-fixer
      - id: trailing-whitespace

  - repo: https://github.com/PyCQA/flake8
    rev: 4.0.1
  - repo: https://github.com/PyCQA/isort
    rev: 5.10.1
    hooks:
      - id: flake8
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/asottile/pyupgrade
    rev: v2.31.0
  - repo: https://github.com/psf/black
    rev: 22.1.0
    hooks:
      - id: pyupgrade
        args: [--py38-plus]
      - id: black

  - repo: https://github.com/myint/autoflake
    rev: v1.4
  - repo: https://github.com/PyCQA/flake8
    rev: 4.0.1
    hooks:
      - id: autoflake
        args:
          - --in-place
          - --remove-unused-variables
          - --remove-all-unused-imports
          - --expand-star-imports
          - --ignore-init-module-imports
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.931
    rev: v0.930
    hooks:
      - id: mypy

  - repo: https://github.com/pycqa/pydocstyle
    rev: 6.1.1
    hooks:
      - id: pydocstyle
        exclude: (test_|tests/|tasks.py)
  13  
CHANGES.rst
@@ -1,8 +1,8 @@
Change log
==========

v2022.2.25
----------
v2022.3.7
---------

* Add VASP WSWQ file parsing, PR #2439 from @jmmshn
* Improve chemical potential diagram plotting, PR #2447 from @mattmcdermott
@@ -13,6 +13,9 @@ v2022.2.25
* Update to ChemEnv citation, PR #2448 from @JaGeo
* Type annotation fix, PR #2432 from @janosh
* Documentation fix for Structure.apply_operation, PR #2433 from @janosh
* Add caching to compatibility classes as speed optimization, PR #2450 from @munrojm

This release was previously intended for v2022.2.25.

Important note: an update to a library that pymatgen depends upon has led to the
~/.pmgrc.yml configuration file being corrupted for many users. If you are affected,
@@ -1587,7 +1590,7 @@ v3.2.5
v3.2.4
------

* GaussianOutput can now parse frequencies, normal modes and cartesian forces
* GaussianOutput can now parse frequencies, normal modes and Cartesian forces
  (Xin Chen).
* Support for Aiida<->pymatgen conversion by the Aiida development team (Andrius
  Merkys).
@@ -1710,7 +1713,7 @@ v3.0.11
v3.0.10
------

* Fix cartesian coord parsing in Poscar class.
* Fix Cartesian coord parsing in Poscar class.
* Vasprun now works with non-GGA PBE runs
* Misc bug fixes

@@ -2119,7 +2122,7 @@ v2.6.3
  PDAnalyzer and PDPlotter in pymatgen.phasediagrams.
* Improvements to StructureMatcher: stol (site - tolerance) redefined as
  a fraction of the average length per atom. Structures matched in fractional
  space are now also matched in cartesian space and a rms displacement
  space are now also matched in Cartesian space and a rms displacement
  normalized by length per atom can be returned using the rms_dist method.

v2.6.2
  6  
docs_rst/change_log.rst
@@ -1570,7 +1570,7 @@ v3.2.5
v3.2.4
------

* GaussianOutput can now parse frequencies, normal modes and cartesian forces
* GaussianOutput can now parse frequencies, normal modes and Cartesian forces
  (Xin Chen).
* Support for Aiida<->pymatgen conversion by the Aiida development team (Andrius
  Merkys).
@@ -1693,7 +1693,7 @@ v3.0.11
v3.0.10
------

* Fix cartesian coord parsing in Poscar class.
* Fix Cartesian coord parsing in Poscar class.
* Vasprun now works with non-GGA PBE runs
* Misc bug fixes

@@ -2102,7 +2102,7 @@ v2.6.3
  PDAnalyzer and PDPlotter in pymatgen.phasediagrams.
* Improvements to StructureMatcher: stol (site - tolerance) redefined as
  a fraction of the average length per atom. Structures matched in fractional
  space are now also matched in cartesian space and a rms displacement
  space are now also matched in Cartesian space and a rms displacement
  normalized by length per atom can be returned using the rms_dist method.

v2.6.2
  2  
docs_rst/introduction.rst
@@ -362,7 +362,7 @@ Here are some quick examples of the core capabilities and objects:
    1 Cl     0.510000     0.510000     0.510000
    2 Cs     0.000000     0.000000     0.000000
    >>>
    >>> # Molecules function similarly, but with Site and cartesian coords.
    >>> # Molecules function similarly, but with Site and Cartesian coords.
    >>> # The following changes the C in CH4 to an N and displaces it by 0.01A
    >>> # in the x-direction.
    >>> methane[0] = "N", [0.01, 0, 0]
  2  
docs_rst/usage.rst
@@ -234,7 +234,7 @@ Molecules. For example, you can change any site simply with::
    molecule[1] = "F"

    # Change species and coordinates (fractional assumed for Structures,
    # cartesian for Molecules)
    # Cartesian for Molecules)
    structure[1] = "Cl", [0.51, 0.51, 0.51]
    molecule[1] = "F", [1.34, 2, 3]

  82  
pymatgen/alchemy/materials.py
@@ -7,20 +7,24 @@
series of transformations.
"""

from __future__ import annotations

import datetime
import json
import os
import re
from typing import Any
from warnings import warn

from monty.json import MontyDecoder, MSONable, jsanitize
from monty.json import MSONable, jsanitize

from pymatgen.alchemy.filters import AbstractStructureFilter
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.sets import MPRelaxSet

dec = MontyDecoder()
from pymatgen.io.vasp.sets import MPRelaxSet, VaspInputSet
from pymatgen.transformations.transformation_abc import AbstractTransformation
from pymatgen.util.provenance import StructureNL


class TransformedStructure(MSONable):
@@ -32,27 +36,33 @@ class TransformedStructure(MSONable):
    associated transformation history.
    """

    def __init__(self, structure, transformations=None, history=None, other_parameters=None):
    def __init__(
        self,
        structure: Structure,
        transformations: list[AbstractTransformation] = None,
        history: list[AbstractTransformation | dict[str, Any]] = None,
        other_parameters: dict[str, Any] = None,
    ) -> None:
        """
        Initializes a transformed structure from a structure.
        Args:
            structure (Structure): Input structure
            transformations ([Transformations]): List of transformations to
            transformations (list[Transformation]): List of transformations to
                apply.
            history (list): Previous history.
            history (list[Transformation]): Previous history.
            other_parameters (dict): Additional parameters to be added.
        """
        self.final_structure = structure
        self.history = history or []
        self.other_parameters = other_parameters or {}
        self._undone = []
        self._undone: list[tuple[AbstractTransformation | dict[str, Any], Structure]] = []

        transformations = transformations or []
        for t in transformations:
            self.append_transformation(t)

    def undo_last_change(self):
    def undo_last_change(self) -> None:
        """
        Undo the last change in the TransformedStructure.
@@ -70,7 +80,7 @@ def undo_last_change(self):
            s = Structure.from_dict(s)
        self.final_structure = s

    def redo_next_change(self):
    def redo_next_change(self) -> None:
        """
        Redo the last undone change in the TransformedStructure.
@@ -83,11 +93,11 @@ def redo_next_change(self):
        self.history.append(h)
        self.final_structure = s

    def __getattr__(self, name):
    def __getattr__(self, name) -> Any:
        s = object.__getattribute__(self, "final_structure")
        return getattr(s, name)

    def __len__(self):
    def __len__(self) -> int:
        return len(self.history)

    def append_transformation(self, transformation, return_alternatives=False, clear_redo=True):
@@ -146,7 +156,7 @@ def append_transformation(self, transformation, return_alternatives=False, clear
        self.final_structure = s
        return None

    def append_filter(self, structure_filter):
    def append_filter(self, structure_filter: AbstractStructureFilter) -> None:
        """
        Adds a filter.
@@ -158,7 +168,9 @@ def append_filter(self, structure_filter):
        hdict["input_structure"] = self.final_structure.as_dict()
        self.history.append(hdict)

    def extend_transformations(self, transformations, return_alternatives=False):
    def extend_transformations(
        self, transformations: list[AbstractTransformation], return_alternatives: bool = False
    ) -> None:
        """
        Extends a sequence of transformations to the TransformedStructure.
@@ -172,7 +184,7 @@ def extend_transformations(self, transformations, return_alternatives=False):
        for t in transformations:
            self.append_transformation(t, return_alternatives=return_alternatives)

    def get_vasp_input(self, vasp_input_set=MPRelaxSet, **kwargs):
    def get_vasp_input(self, vasp_input_set: type[VaspInputSet] = MPRelaxSet, **kwargs) -> dict[str, Any]:
        """
        Returns VASP input as a dict of vasp objects.
@@ -184,7 +196,13 @@ def get_vasp_input(self, vasp_input_set=MPRelaxSet, **kwargs):
        d["transformations.json"] = json.dumps(self.as_dict())
        return d

    def write_vasp_input(self, vasp_input_set=MPRelaxSet, output_dir=".", create_directory=True, **kwargs):
    def write_vasp_input(
        self,
        vasp_input_set: type[VaspInputSet] = MPRelaxSet,
        output_dir: str = ".",
        create_directory: bool = True,
        **kwargs,
    ) -> None:
        """
        Writes VASP input to an output_dir.
@@ -201,7 +219,7 @@ def write_vasp_input(self, vasp_input_set=MPRelaxSet, output_dir=".", create_dir
        with open(os.path.join(output_dir, "transformations.json"), "w") as fp:
            json.dump(self.as_dict(), fp)

    def __str__(self):
    def __str__(self) -> str:
        output = [
            "Current structure",
            "------------",
@@ -217,7 +235,7 @@ def __str__(self):
        output.append(str(self.other_parameters))
        return "\n".join(output)

    def set_parameter(self, key, value):
    def set_parameter(self, key: str, value: Any) -> None:
        """
        Set a parameter
@@ -227,7 +245,7 @@ def set_parameter(self, key, value):
        self.other_parameters[key] = value

    @property
    def was_modified(self):
    def was_modified(self) -> bool:
        """
        Boolean describing whether the last transformation on the structure
        made any alterations to it one example of when this would return false
@@ -237,7 +255,7 @@ def was_modified(self):
        return not self.final_structure == self.structures[-2]

    @property
    def structures(self):
    def structures(self) -> list[Structure]:
        """
        Copy of all structures in the TransformedStructure. A
        structure is stored after every single transformation.
@@ -246,15 +264,20 @@ def structures(self):
        return hstructs + [self.final_structure]

    @staticmethod
    def from_cif_string(cif_string, transformations=None, primitive=True, occupancy_tolerance=1.0):
    def from_cif_string(
        cif_string: str,
        transformations: list[AbstractTransformation] = None,
        primitive: bool = True,
        occupancy_tolerance: float = 1.0,
    ) -> TransformedStructure:
        """
        Generates TransformedStructure from a cif string.
        Args:
            cif_string (str): Input cif string. Should contain only one
                structure. For cifs containing multiple structures, please use
                CifTransmuter.
            transformations ([Transformations]): Sequence of transformations
            transformations (list[Transformation]): Sequence of transformations
                to be applied to the input structure.
            primitive (bool): Option to set if the primitive cell should be
                extracted. Defaults to True. However, there are certain
@@ -287,13 +310,15 @@ def from_cif_string(cif_string, transformations=None, primitive=True, occupancy_
        return TransformedStructure(s, transformations, history=[source_info])

    @staticmethod
    def from_poscar_string(poscar_string, transformations=None):
    def from_poscar_string(
        poscar_string: str, transformations: list[AbstractTransformation] = None
    ) -> TransformedStructure:
        """
        Generates TransformedStructure from a poscar string.
        Args:
            poscar_string (str): Input POSCAR string.
            transformations ([Transformations]): Sequence of transformations
            transformations (list[Transformation]): Sequence of transformations
                to be applied to the input structure.
        """
        p = Poscar.from_string(poscar_string)
@@ -310,7 +335,7 @@ def from_poscar_string(poscar_string, transformations=None):
        }
        return TransformedStructure(s, transformations, history=[source_info])

    def as_dict(self):
    def as_dict(self) -> dict[str, Any]:
        """
        Dict representation of the TransformedStructure.
        """
@@ -323,14 +348,14 @@ def as_dict(self):
        return d

    @classmethod
    def from_dict(cls, d):
    def from_dict(cls, d) -> TransformedStructure:
        """
        Creates a TransformedStructure from a dict.
        """
        s = Structure.from_dict(d)
        return cls(s, history=d["history"], other_parameters=d.get("other_parameters", None))

    def to_snl(self, authors, **kwargs):
    def to_snl(self, authors, **kwargs) -> StructureNL:
        """
        Generate SNL from TransformedStructure.
@@ -350,12 +375,11 @@ def to_snl(self, authors, **kwargs):
                    "description": h,
                }
            )
        from pymatgen.util.provenance import StructureNL

        return StructureNL(self.final_structure, authors, history=hist, **kwargs)

    @classmethod
    def from_snl(cls, snl):
    def from_snl(cls, snl: StructureNL) -> TransformedStructure:
        """
        Create TransformedStructure from SNL.
  8  
pymatgen/analysis/adsorption.py
@@ -238,7 +238,7 @@ def find_adsorption_sites(
    ):
        """
        Finds surface sites according to the above algorithm.  Returns
        a list of corresponding cartesian coordinates.
        a list of corresponding Cartesian coordinates.
        Args:
            distance (float): distance from the coordinating ensemble
@@ -316,7 +316,7 @@ def symm_reduce(self, coords_set, threshold=1e-6):
        symmetrically equivalent duplicates
        Args:
            coords_set: coordinate set in cartesian coordinates
            coords_set: coordinate set in Cartesian coordinates
            threshold: tolerance for distance equivalence, used
                as input to in_coord_list_pbc for dupl. checking
        """
@@ -364,7 +364,7 @@ def ensemble_center(cls, site_list, indices, cartesian=True):
            indices (list of ints): list of ints from which to select
                sites from site list
            cartesian (bool): whether to get average fractional or
                cartesian coordinate
                Cartesian coordinate
        """
        if cartesian:
            return np.average([site_list[i].coords for i in indices], axis=0)
@@ -639,7 +639,7 @@ def get_rot(slab):

def put_coord_inside(lattice, cart_coordinate):
    """
    converts a cartesian coordinate such that it is inside the unit cell.
    converts a Cartesian coordinate such that it is inside the unit cell.
    """
    fc = lattice.get_fractional_coords(cart_coordinate)
    return lattice.get_cartesian_coords([c - np.floor(c) for c in fc])
  2  
pymatgen/analysis/chemenv/coordination_environments/coordination_geometry_finder.py
@@ -498,7 +498,7 @@ def set_structure(self, lattice, species, coords, coords_are_cartesian):
        :param lattice: The lattice of the structure
        :param species: The species on the sites
        :param coords: The coordinates of the sites
        :param coords_are_cartesian: If set to True, the coordinates are given in cartesian coordinates
        :param coords_are_cartesian: If set to True, the coordinates are given in Cartesian coordinates
        """
        self.setup_structure(Structure(lattice, species, coords, coords_are_cartesian))

  2  
pymatgen/analysis/chemenv/utils/math_utils.py
@@ -20,7 +20,7 @@
from scipy.special import erf

##############################################################
# cartesian product of lists ##################################
# Cartesian product of lists ##################################
##############################################################


  87  
pymatgen/analysis/chempot_diagram.py
@@ -22,12 +22,13 @@
    258 (1999). https://doi.org/10.1361/105497199770335794
"""

from __future__ import annotations

import json
import os
import warnings
from functools import lru_cache
from itertools import groupby
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.express as px
@@ -65,8 +66,8 @@ class ChemicalPotentialDiagram(MSONable):

    def __init__(
        self,
        entries: List[PDEntry],
        limits: Optional[Dict[Element, float]] = None,
        entries: list[PDEntry],
        limits: dict[Element, float] | None = None,
        default_min_limit: float = -50.0,
    ):
        """
@@ -106,13 +107,13 @@ def __init__(

    def get_plot(
        self,
        elements: Optional[List[Union[Element, str]]] = None,
        label_stable: Optional[bool] = True,
        formulas_to_draw: Optional[List[str]] = None,
        draw_formula_meshes: Optional[bool] = True,
        draw_formula_lines: Optional[bool] = True,
        formula_colors: List[str] = px.colors.qualitative.Dark2,
        element_padding: Optional[float] = 1.0,
        elements: list[Element | str] | None = None,
        label_stable: bool | None = True,
        formulas_to_draw: list[str] | None = None,
        draw_formula_meshes: bool | None = True,
        draw_formula_lines: bool | None = True,
        formula_colors: list[str] = px.colors.qualitative.Dark2,
        element_padding: float | None = 1.0,
    ) -> Figure:
        """
        Plot the 2-dimensional or 3-dimensional chemical potential diagram using an
@@ -178,7 +179,7 @@ def get_plot(

        return fig

    def _get_domains(self) -> Dict[str, np.ndarray]:
    def _get_domains(self) -> dict[str, np.ndarray]:
        """Returns a dictionary of domains as {formula: np.ndarray}"""
        hyperplanes = self._hyperplanes
        border_hyperplanes = self._border_hyperplanes
@@ -211,7 +212,7 @@ def _get_border_hyperplanes(self) -> np.ndarray:

        return border_hyperplanes

    def _get_hyperplanes_and_entries(self) -> Tuple[np.ndarray, List[PDEntry]]:
    def _get_hyperplanes_and_entries(self) -> tuple[np.ndarray, list[PDEntry]]:
        """
        Returns both the array of hyperplanes, as well as a list of the minimum
        entries.
@@ -235,9 +236,7 @@ def _get_hyperplanes_and_entries(self) -> Tuple[np.ndarray, List[PDEntry]]:

        return hyperplanes, hyperplane_entries

    def _get_2d_plot(
        self, elements: List[Element], label_stable: Optional[bool], element_padding: Optional[float]
    ) -> Figure:
    def _get_2d_plot(self, elements: list[Element], label_stable: bool | None, element_padding: float | None) -> Figure:
        """Returns a Plotly figure for a 2-dimensional chemical potential diagram"""
        domains = self.domains.copy()
        elem_indices = [self.elements.index(e) for e in elements]
@@ -285,13 +284,13 @@ def _get_2d_plot(

    def _get_3d_plot(
        self,
        elements: List[Element],
        label_stable: Optional[bool],
        formulas_to_draw: Optional[List[str]],
        draw_formula_meshes: Optional[bool],
        draw_formula_lines: Optional[bool],
        formula_colors: Optional[List[str]],
        element_padding: Optional[float],
        elements: list[Element],
        label_stable: bool | None,
        formulas_to_draw: list[str] | None,
        draw_formula_meshes: bool | None,
        draw_formula_lines: bool | None,
        formula_colors: list[str] | None,
        element_padding: float | None,
    ) -> Figure:
        """Returns a Plotly figure for a 3-dimensional chemical potential diagram."""

@@ -301,8 +300,8 @@ def _get_3d_plot(
        elem_indices = [self.elements.index(e) for e in elements]

        domains = self.domains.copy()
        domain_simplexes: Dict[str, Optional[List[Simplex]]] = {}
        draw_domains: Dict[str, np.ndarray] = {}
        domain_simplexes: dict[str, list[Simplex] | None] = {}
        draw_domains: dict[str, np.ndarray] = {}
        draw_comps = [Composition(formula).reduced_composition for formula in formulas_to_draw]
        annotations = []

@@ -375,8 +374,8 @@ def _get_3d_plot(

    @staticmethod
    def _get_new_limits_from_padding(
        domains: Dict[str, np.ndarray],
        elem_indices: List[int],
        domains: dict[str, np.ndarray],
        elem_indices: list[int],
        element_padding: float,
        default_min_limit: float,
    ):
@@ -395,7 +394,7 @@ def _get_new_limits_from_padding(
        return new_lims

    @staticmethod
    def _get_2d_domain_lines(draw_domains) -> List[Scatter]:
    def _get_2d_domain_lines(draw_domains) -> list[Scatter]:
        """
        Returns a list of Scatter objects tracing the domain lines on a
        2-dimensional chemical potential diagram.
@@ -418,7 +417,7 @@ def _get_2d_domain_lines(draw_domains) -> List[Scatter]:
        return lines

    @staticmethod
    def _get_3d_domain_lines(domains: Dict[str, Optional[List[Simplex]]]) -> List[Scatter3d]:
    def _get_3d_domain_lines(domains: dict[str, list[Simplex] | None]) -> list[Scatter3d]:
        """
        Returns a list of Scatter3d objects tracing the domain lines on a
        3-dimensional chemical potential diagram.
@@ -446,7 +445,7 @@ def _get_3d_domain_lines(domains: Dict[str, Optional[List[Simplex]]]) -> List[Sc
    @staticmethod
    def _get_3d_domain_simplexes_and_ann_loc(
        points_3d: np.ndarray,
    ) -> Tuple[List[Simplex], np.ndarray]:
    ) -> tuple[list[Simplex], np.ndarray]:
        """
        Returns a list of Simplex objects and coordinates of annotation for one
        domain in a 3-d chemical potential diagram. Uses PCA to project domain
@@ -464,9 +463,9 @@ def _get_3d_domain_simplexes_and_ann_loc(

    @staticmethod
    def _get_3d_formula_meshes(
        draw_domains: Dict[str, np.ndarray],
        formula_colors: Optional[List[str]],
    ) -> List[Mesh3d]:
        draw_domains: dict[str, np.ndarray],
        formula_colors: list[str] | None,
    ) -> list[Mesh3d]:
        """
        Returns a list of Mesh3d objects for the domains specified by the
        user (i.e., draw_domains).
@@ -493,9 +492,9 @@ def _get_3d_formula_meshes(

    @staticmethod
    def _get_3d_formula_lines(
        draw_domains: Dict[str, np.ndarray],

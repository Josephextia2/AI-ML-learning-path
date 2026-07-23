# 2D Gauss Law (Laplace equation and Gauss equation)
import os
import warnings
#import subprocess
from pathlib import Path

import sympy as sp
from sympy import Symbol, Function, Heaviside
import numpy as np

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.key import Key
from physicsnemo.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,
)

from physicsnemo.sym.amp import AmpManager
from physicsnemo.sym.eq.pde import PDE

#epsilon_0 = 8.854187817e-12  # vacuum permittivity in F/m
#electron_charge = 1.602176634e-19  # elementary charge in C. Take positive sign

# hyperparameters
height, width = 1, 1
x0, y0 = -width / 2, -height / 2

charge_radius = 0.1
point_charges_positions= np.array([[0.3, 0.3], [0.7, 0.7]]) + np.array([[x0, y0]])      # dim: [2, 2] + [1, 2] -> [2, 2]
charges = np.array([2, -2])                                                             # charge to epsilon ratio instead of charge. 

root_dir = os.getcwd()
Path(root_dir + "/conf").mkdir(parents=True, exist_ok=True)
with open(root_dir + "/conf/config.yaml", 'w') as f:
    f.write(
        f"""
        defaults:
            - physicsnemo_default
            - arch:
                - fully_connected
            - scheduler: tf_exponential_lr
            - optimizer: lamb
            - loss: sum
            - _self_
        
        scheduler:
            decay_rate: 0.95
            decay_steps: 4000
        
        training:
            rec_validation_freq: 1000
            rec_inference_freq: 2000
            rec_monitor_freq: 1000
            rec_constraint_freq: 2000
            max_steps: 50000

        batch_size:
            SideBoundary: 4000
            TopBottomBoundary: 4000
            Interior: 9000

        graph:
            func_arch: true
        """
    )

class GaussEquation2D(PDE):
    name = "GaussEquation2D"

    def __init__(self):
        # coordinates
        x, y = Symbol("x"), Symbol("y")

        input_variables ={"x": x, "y": y}

        # electric potential function
        v = Function("v")(*input_variables)

        rho = 0
        for i in range(len(charges)):
            rho += charges[i] / (sp.pi * charge_radius**2) * (Heaviside(charge_radius - sp.sqrt((x - point_charges_positions[i, 0])**2 + (y - point_charges_positions[i, 1])**2)))

        # set equation
        self.equations = {}
        self.equations["gauss"] = v.diff(x, 2) + v.diff(y, 2) + rho         # charge density to epsilon ratio instead of charge density.

@physicsnemo.sym.main(config_path=root_dir + "/conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    diff_equation = GaussEquation2D()
    potential_net = instantiate_arch(
        input_keys = [Key("x"), Key("y")], 
        output_keys = [Key("v")],
        cfg = cfg.arch.fully_connected,
    )
    nodes = diff_equation.make_nodes() + [potential_net.make_node(name="potential_network")]

    # add constraints to solver
    # make geometry
    x, y = Symbol("x"), Symbol("y")
    rec = Rectangle((x0, y0), (x0 + width, y0 + height))

    # make domain
    domain = Domain()

    # initial condition
    interior = PointwiseInteriorConstraint(
        nodes = nodes, 
        geometry = rec,
        outvar = {"gauss": 0},
        batch_size = cfg.batch_size.Interior,
    )
    domain.add_constraint(interior, "interior")

    side_boundary = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = rec,
        outvar = {"v": 0.0},
        batch_size = cfg.batch_size.SideBoundary,
        criteria = ((y < height / 2) & (y > -height / 2))
    )
    domain.add_constraint(side_boundary, "side_boundary")

    top_bottom_boundary = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = rec,
        outvar = {"v": 0.8},
        batch_size = cfg.batch_size.TopBottomBoundary,
        criteria = (x < width / 2) & (x > -width / 2)
    )
    domain.add_constraint(top_bottom_boundary, "top_bottom_boundary")


    # add validator
    file_path = "2point_charges-Jacobi.csv"
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {"x": "x", "y": "y", "v": "v"}
        var = csv_to_dict(to_absolute_path(file_path), mapping)
        invar_numpy = {key: value for key, value in var.items() if key in ["x", "y"]}
        outvar_numpy = {key: value for key, value in var.items() if key in ["v"]}

        validator = PointwiseValidator(
            nodes = nodes,
            invar = invar_numpy,
            true_outvar = outvar_numpy,
            batch_size = 1024,
            plotter = ValidatorPlotter(),
        )
        domain.add_validator(validator)

        # add inferencer
        inferencer = PointwiseInferencer(
            nodes = nodes,
            invar = invar_numpy,
            output_names = ["v"],
            batch_size = 1024,
            plotter = InferencerPlotter(),
        )
        domain.add_inferencer(inferencer, "inf_data")

    else:
        warnings.warn(f"Directory {file_path} does not exist.")
        
    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()

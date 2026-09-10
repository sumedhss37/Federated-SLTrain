"""Optional future Muon integration point.

The rest of the project does not import a Muon package. When you are ready,
provide a torch-style optimizer class/factory here and switch the server to
OptimizerAggregator. This keeps the federated protocol independent of the
server optimizer.
"""

from .aggregators import OptimizerAggregator, MultiOptimizerAggregator


def make_muon_server_aggregator(global_state, MuonOptimizerClass, lr, weight_decay=0.0):
    """Adapter for a Muon optimizer implementation with a torch-style API.

    Example (adjust constructor args to the Muon implementation you install):
        from your_muon_package import Muon
        aggregator = make_muon_server_aggregator(
            global_state, Muon, lr=0.02, weight_decay=0.0
        )

    For optimizers that only support matrices, add a parameter_filter that
    selects L/R (or other 2-D parameters) and use a second optimizer for the
    remaining vectors/scalars.
    """
    return OptimizerAggregator(
        global_state,
        optimizer_factory=MuonOptimizerClass,
        lr=lr,
        weight_decay=weight_decay,
    )


def make_split_muon_server_aggregator(
    global_state,
    MuonOptimizerClass,
    SGDOptimizerClass,
    muon_lr=0.02,
    fallback_lr=1.0,
):
    """Use Muon for 2-D matrix parameters and SGD for everything else.

    Rename/adjust the predicate when integrating the particular Muon package
    you choose."""
    return MultiOptimizerAggregator(
        global_state,
        groups=[
            (MuonOptimizerClass, lambda n, p: p.ndim == 2, muon_lr, 0.0),
            (SGDOptimizerClass, lambda n, p: p.ndim != 2, fallback_lr, 0.0),
        ],
    )

import difflib
from copy import deepcopy
from typing import Any, Callable, NamedTuple, Optional, Tuple

import torch
import torch.utils._pytree as pytree
from torch.export import ExportedProgram
from torch.utils._pytree import tree_flatten


class ErrorLimits(NamedTuple):
    """Limits used for numerical checks between tensors"""

    rtol: Optional[float] = None
    atol: Optional[float] = None


class GraphAndEagerModuleOutputDiff:
    def __init__(self, gm, inputs, module_output, gm_output, diff_reason):
        self.gm = gm
        self.inputs = inputs
        self.module_output = module_output
        self.gm_output = gm_output
        self.diff_reason = diff_reason

    def __str__(self):
        return self.diff_reason

    def __repr__(self):
        return self.diff_reason


class OutputCheckResult:
    def __init__(self, is_same, reasons):
        self.is_same = is_same
        self.reasons = reasons

    def __str__(self):
        return "Is Same: {}, reasons: {}".format(self.is_same, self.reasons)

    def __repr__(self):
        return "Is Same: {}, reasons: {}".format(self.is_same, self.reasons)


class GraphModuleDiff:
    def __init__(self, gm_orig, gm_new, inputs, tracing_inputs):
        self.gm_orig = gm_orig
        self.gm_new = gm_new
        self.inputs = inputs
        self.tracing_inputs = tracing_inputs
        gm_lines = str(gm_orig).split("\n")
        gm_new_lines = str(gm_new).split("\n")

        # The difference between the 2 graph modules as a unified diff string.
        self.gm_diff = "\n".join(
            list(
                difflib.unified_diff(
                    gm_lines,
                    gm_new_lines,
                    fromfile="first_graph_module.py",
                    tofile="subsequent_graph_module.py",
                )
            )
        )

    def __str__(self):
        return self.gm_diff


def check_numerical(
    graph_lhs: ExportedProgram,
    graph_rhs: ExportedProgram,
    representative_input,
    limits: ErrorLimits,
) -> Optional[GraphAndEagerModuleOutputDiff]:
    """
    Run the input representative_input through both graphs
    (graph_lhs and graph_rhs) and check if the outputs are close enough
    """

    representative_input = pytree.tree_map(
        lambda arg: arg.detach() if isinstance(arg, torch.Tensor) else arg,
        representative_input,
    )
    inputs_lhs = deepcopy(representative_input)
    inputs_rhs = deepcopy(representative_input)

    if isinstance(graph_lhs, ExportedProgram):
        graph_lhs = graph_lhs.module()

    if isinstance(graph_rhs, ExportedProgram):
        graph_rhs = graph_rhs.module()

    output_lhs = graph_lhs(*inputs_lhs)
    output_rhs = graph_rhs(*inputs_rhs)

    # Check if eager_output and gm_output are close enough.
    reason = check_close_enough("", output_lhs, output_rhs, limits)
    if reason is not None:
        inputs_copy = deepcopy(representative_input)
        return GraphAndEagerModuleOutputDiff(
            graph_rhs, inputs_copy, output_lhs, output_rhs, reason
        )

    return None



def check_close_enough(  # noqa flake8 C901 too complex
    prefix,
    lhs,
    rhs,
    limits: ErrorLimits,
) -> Optional[str]:
    if type(lhs) != type(rhs):
        return "{} type(lhs) == {}, type(rhs) == {}".format(
            prefix, type(lhs), type(rhs)
        )
    if isinstance(lhs, tuple):
        if len(lhs) != len(rhs):
            return "{} !Tuple! len(lhs) == {}, len(rhs) == {}".format(
                prefix, len(lhs), len(rhs)
            )
        for i in range(len(lhs)):
            ret = check_close_enough(
                prefix + ("({})".format(i)), lhs[i], rhs[i], limits
            )
            if ret is not None:
                return ret
    elif isinstance(lhs, list):
        if len(lhs) != len(rhs):
            return "{} !List! len(lhs) == {}, len(rhs) == {}".format(
                prefix, len(lhs), len(rhs)
            )
        for i in range(len(lhs)):
            ret = check_close_enough(
                prefix + ("[{}]".format(i)), lhs[i], rhs[i], limits
            )
            if ret is not None:
                return ret
    elif isinstance(lhs, dict):
        if len(lhs) != len(rhs):
            return "{} !Dict! len(lhs) == {}, len(rhs) == {}".format(
                prefix, len(lhs), len(rhs)
            )
        lhs_keys = lhs.keys()
        rhs_keys = rhs.keys()

        if lhs_keys != rhs_keys:
            return prefix + " Dict Keys Unequal"
        for k in lhs.keys():
            ret = check_close_enough(
                prefix + "{{{}}}".format(k), lhs[k], rhs[k], limits
            )
            if ret is not None:
                return ret
    elif isinstance(lhs, torch.Tensor):
        try:
            torch.testing.assert_close(rhs, lhs, rtol=limits.rtol, atol=limits.atol)
        except Exception as ex:
            return prefix + " !Tensor! " + str(ex)
    elif isinstance(lhs, int) or isinstance(lhs, float) or isinstance(lhs, str):
        if lhs != rhs:
            return "{} LHS == {}, RHS == {}".format(prefix, lhs, rhs)
    elif lhs is None:
        if rhs is not None:
            return f"LHS is None, RHS is {rhs} ({type(rhs)})"
    else:
        raise TypeError("Unhandled type: {}".format(type(lhs)))
    return None


@torch.no_grad()
def check_executorch_output_consistency(
    flatbuffer_buff: bytes,
    model: torch.nn.Module,
    method_name: str,
    example_inputs: Tuple[torch.Tensor, ...],
    load_fn: Callable,
    error_limits: ErrorLimits,
):
    reasons = []
    is_same = True

    executorch_m = load_fn(flatbuffer_buff)
    eager_m = model
    # Put eager model in eval mode
    eager_m.eval()

    # Flatten input
    inputs_flattened, _ = tree_flatten(example_inputs)

    # Gather some eager output as a basis of comparison
    eager_fn = getattr(eager_m, method_name)
    eager_output = eager_fn(*example_inputs)

    # Flatten the eager output so we can compare it to the flat
    # executorch output. One could also un-flatten the executorch
    # output using the pytree schema of the eager output and then
    # compare.
    flat_eager_output, _ = tree_flatten(eager_output)

    # Execute the Executorch program with the same input, output is pre-flattened
    executorch_output = executorch_m.run_method(
        method_name, tuple(inputs_flattened)
    )

    reason = check_close_enough(
        method_name, flat_eager_output, executorch_output, error_limits
    )
    if reason is not None:
        is_same = False
        reasons.append(reason)

    return OutputCheckResult(is_same, reasons)

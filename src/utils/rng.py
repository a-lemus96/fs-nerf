# stdlib modules
import contextlib

# third-party modules
import torch


@contextlib.contextmanager
def use_generator(generator: torch.Generator):
    """
    Temporarily routes the global torch RNG through a dedicated generator,
    for call chains with no exposed generator= parameter (e.g. nn.Module
    weight init). Saves the advanced state back into the generator on exit
    and restores whatever was globally active beforehand, so the stream
    keeps advancing across repeated calls instead of replaying the same
    draws each time.

    Args:
        generator (torch.Generator): dedicated stream driving calls made
            inside this context; its device.type selects the CPU/CUDA path.
    """
    if generator.device.type == "cuda":
        device = generator.device
        outer_state = torch.cuda.get_rng_state(device)
        torch.cuda.set_rng_state(generator.get_state(), device)
        try:
            yield
        finally:
            generator.set_state(torch.cuda.get_rng_state(device))
            torch.cuda.set_rng_state(outer_state, device)
    else:
        outer_state = torch.get_rng_state()
        torch.set_rng_state(generator.get_state())
        try:
            yield
        finally:
            generator.set_state(torch.get_rng_state())
            torch.set_rng_state(outer_state)

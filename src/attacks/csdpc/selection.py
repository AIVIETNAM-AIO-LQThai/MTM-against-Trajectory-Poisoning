"""CSDPC rare-pattern and target-pattern selection."""


def select_source_patterns(*args, **kwargs):
    raise NotImplementedError(
        "Rare-pattern selection is blocked by source audit."
    )


def select_target_patterns(*args, **kwargs):
    raise NotImplementedError(
        "Target-pattern selection is blocked by source audit."
    )
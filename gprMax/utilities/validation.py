"""Shared validation for public keyword-based model definitions."""

from difflib import get_close_matches


def validate_keywords(kwargs, allowed, *, context):
    """Reject unknown keys, even when their values are None, False, or zero.

    Never correct a spelling automatically: a suggested option may change
    the physics of the model and must be chosen explicitly by the caller.
    """
    unknown = sorted(set(kwargs) - set(allowed))
    if not unknown:
        return
    details = []
    for name in unknown:
        matches = get_close_matches(name, sorted(allowed), n=1, cutoff=0.7)
        hint = f" (did you mean {matches[0]!r}?)" if matches else ""
        details.append(f"{name!r}{hint}")
    raise TypeError(
        f"{context} got unexpected keyword argument(s): {', '.join(details)}. "
        f"Accepted keywords: {', '.join(sorted(allowed))}."
    )

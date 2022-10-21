# TODO move somewhere else
#  (for now i want to keep changes in refactor_npc branch contained to tenpy.linalg as much as possible

def force_str_len(obj, length: int, rjust: bool = True, placeholder: str = '[...]') -> str:
    """Convert an object to a string, then force the string to a given length.
    If it is too short, right (rjust=True) or left (rjust=False) justify it.
    If it is too long, replace a central portion with the placeholder.
    """
    assert length >= 0
    obj = str(obj)
    if len(obj) <= length:
        return obj.rjust(length) if rjust else obj.ljust(length)
    else:
        num_chars = length - len(placeholder)
        assert num_chars >= 0, f'Placeholder {placeholder} is longer than length={length}!'
        left_chars = num_chars // 2
        right_chars = num_chars - left_chars
        return obj[:left_chars] + placeholder + obj[-right_chars:]

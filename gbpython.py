"""Collection of auxiliary functions.

"""

def skipcomment(f, char='#'):
    """Skip comment lines and empty lines when reading a file
    and return the first non-comment, non-empty line.

    f: file handle
    char: character denoting a comment line, e.g. '#'
    """
    while True:
        line = f.readline()
        linesplit = line.split()
        # EOF
        if not line:
            break
        # comment or empty line
        elif line.startswith(char) or not linesplit:
            continue
        else:
            break
    return line


import os



from pathlib import Path
import sys


path = Path(__file__)
project_root= str(path.parent.absolute())

if __name__ == 'main':
    print(parent_root, file=sys.stderr)
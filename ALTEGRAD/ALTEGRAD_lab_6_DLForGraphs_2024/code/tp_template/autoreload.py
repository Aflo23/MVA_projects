import sys
from pathlib import Path

from IPython.core.getipython import get_ipython

def is_executed_within_notebook() -> bool:
    ipython = get_ipython()

    if ipython:
        shell_name = ipython.__class__.__name__
        return shell_name == 'ZMQInteractiveShell'
    
    else:
        return False
    
def autoreload_modules() -> None:
    print("Autoreload activated")

    ipython = get_ipython()
    ipython.magic('load_ext autoreload') 
    ipython.magic('autoreload 2')


def autoreload_if_notebook() -> None:
    if is_executed_within_notebook():
        autoreload_modules()


def load_custom_library_with_path(path:Path)-> None:
    sys.path.append(str(path))

# from IPython import get_ipython

# def autoreload_if_notebook() -> None:
#     if is_executed_within_notebook():
#         autoreload_modules()

# def autoreload_modules() -> None:
#     print("Autoreload activated")
#     ipython = get_ipython()
#     if ipython is not None:
#         ipython.magic('load_ext autoreload')
#         ipython.magic('autoreload 2')
#     else:
#         print("This function can only be run in an IPython environment.")

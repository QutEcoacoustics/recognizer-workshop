#!/usr/bin/env python3
import os
import ast
import argparse
import fnmatch
from typing import List, Set, Tuple, Dict, Any

# --- Configuration for files and directories to exclude ---
# This list mimics the behavior of the provided `tree` command.
EXCLUSION_PATTERNS = [
    'cache', '__pycache__', '.git', '*.pyc', '*.pyo',
    '.pytest_cache', '*.parquet', 'venv', 'dist', 'build',
    'local_scripts', 'perch-hoplite'
]

def get_return_value_repr(node: ast.Return) -> str:
    """
    Generates a string representation of the value in a return statement.
    Uses ast.unparse if available (Python 3.9+), otherwise falls back to
    a simpler representation for common types.
    """
    if node.value is None:
        return "None"
    
    # ast.unparse is the most reliable way to convert a node back to source
    if hasattr(ast, 'unparse'):
        return ast.unparse(node.value).strip()
    
    # Fallback for older Python versions
    if isinstance(node.value, ast.Constant):
        return repr(node.value.value)
    if isinstance(node.value, ast.Name):
        return node.value.id
    if isinstance(node.value, ast.Tuple):
        return f"({', '.join(el.id for el in node.value.elts if isinstance(el, ast.Name))})"
    
    return "..." # Placeholder for complex return types on older Python

class FunctionVisitor(ast.NodeVisitor):
    """
    An AST visitor that traverses the code to find functions and their
    return statements. It handles nested functions and methods within classes.
    """
    def __init__(self):
        # Stores the hierarchy, e.g., ['MyClass', 'my_method']
        self.path: List[str] = []
        # Stores the final results: Dict[function_name, set_of_return_values]
        self.functions: Dict[str, Set[str]] = {}

    def _visit_function(self, node: ast.AST) -> None:
        """Helper to handle both FunctionDef and AsyncFunctionDef."""
        # Record current function in the path
        self.path.append(node.name)
        
        # Create a key for the results dictionary, showing nesting
        full_name = ' -> '.join(self.path)
        self.functions[full_name] = set()

        # Look for return statements within this function's body
        for body_item in node.body:
            for child in ast.walk(body_item):
                if isinstance(child, ast.Return):
                    return_repr = get_return_value_repr(child)
                    self.functions[full_name].add(return_repr)
        
        # Continue traversal for nested functions
        self.generic_visit(node)
        
        # Done with this function, pop it from the path
        self.path.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit methods inside a class definition."""
        self.path.append(node.name)
        self.generic_visit(node)
        self.path.pop()

def analyze_python_file(filepath: str) -> Dict[str, Set[str]]:
    """
    Parses a single Python file and returns a dictionary of its
    functions and their return values.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source_code = f.read()
        tree = ast.parse(source_code, filename=filepath)
        visitor = FunctionVisitor()
        visitor.visit(tree)
        return visitor.functions
    except (SyntaxError, UnicodeDecodeError) as e:
        # Return an error message if the file can't be parsed
        return {"Error parsing file": {f"({type(e).__name__}: {e})" }}
    except Exception as e:
        return {"Unexpected error": {f"({type(e).__name__}: {e})" }}


def is_excluded(name: str, patterns: List[str]) -> bool:
    """Check if a file or directory name matches any exclusion pattern."""
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)

def generate_tree(root_dir: str, patterns: List[str]) -> None:
    """
    Walks through a directory, printing a tree structure of files and folders,
    and analyzes any Python files found.
    """
    print(f"\033[1;34m{os.path.basename(os.path.abspath(root_dir))}\033[0m") # Print root in blue

    for root, dirs, files in os.walk(root_dir, topdown=True):
        # Prune directories by modifying the list in-place
        dirs[:] = [d for d in dirs if not is_excluded(d, patterns)]
        
        # Filter out excluded files
        files = [f for f in files if not is_excluded(f, patterns)]

        level = root.replace(root_dir, '').count(os.sep)
        indent = '│   ' * level
        
        # Sort for consistent output
        dirs.sort()
        files.sort()
        
        entries = dirs + files
        for i, name in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = '└── ' if is_last else '├── '
            
            # Use colors for better readability
            is_dir = os.path.isdir(os.path.join(root, name))
            color_code = '\033[1;34m' if is_dir else '' # Blue for dirs
            reset_code = '\033[0m' if color_code else ''

            print(f"{indent}{connector}{color_code}{name}{reset_code}")

            # If it's a Python file, analyze and print its functions
            if name.endswith('.py') and not is_dir:
                filepath = os.path.join(root, name)
                functions = analyze_python_file(filepath)
                
                func_indent = indent + ('    ' if is_last else '│   ')
                
                if not functions:
                    print(f"{func_indent}\033[3m(No functions found)\033[0m")
                    continue

                for func_name, returns in sorted(functions.items()):
                    # Use a different color/style for function info
                    func_line = f"\033[32m* {func_name}\033[0m" # Green
                    if returns:
                        return_str = ", ".join(sorted(list(returns)))
                        func_line += f": \033[3mReturns ({return_str})\033[0m"
                    else:
                        func_line += ": \033[3m(No return statements)\033[0m"
                    print(f"{func_indent}{func_line}")


def main():
    """Main function to parse arguments and start the tree generation."""
    parser = argparse.ArgumentParser(
        description="""
        Lists files and directories in a tree structure, excluding certain
        patterns. For each Python file, it also lists all functions and
        their return values.
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='The root directory to start the tree from (default: current directory).'
    )
    args = parser.parse_args()

    start_path = os.path.abspath(args.directory)
    if not os.path.isdir(start_path):
        print(f"Error: Directory not found at '{start_path}'")
        return

    generate_tree(start_path, EXCLUSION_PATTERNS)

if __name__ == '__main__':
    main()

import ast
import os
from pprint import pprint as print
import sys

class Visitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []         # [[fn_nane1, docstring], ......]
        self.classes = []           # [[[class_name1, docstring], [[method1, docstring], [method2, docstring],....]], [...], ...]
        self.depth = 0
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.depth == 1:     # it's inside a class, last elem in self.classes is the specified class
            class_data = self.classes[-1]
            method_data = class_data[1]

            method_data.append([node.name, ast.get_docstring(node)])

        else:
            self.functions.append([node.name, ast.get_docstring(node)])

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if self.depth == 1:     # it's inside a class, last elem in self.classes is the specified class
            class_data = self.classes[-1]
            method_data = class_data[1]

            method_data.append([node.name, ast.get_docstring(node)])
        else:
            self.functions.append([node.name, ast.get_docstring(node)])
    
    def visit_ClassDef(self, node: ast.ClassDef):
        self.classes.append([[node.name, ast.get_docstring(node)], []])
        self.depth += 1

        for item in node.body:
            self.visit(item)
        
        self.depth -= 1


def source_code_metadata_generator(filepath: str):
    """
    Generates function/class name, and the associated docstrings with them.
    """
    
    with open(filepath, "r") as f:
        source_code = f.read()

    tree = ast.parse(source_code)
    visitor = Visitor()
    visitor.visit(tree)

    return visitor.functions, visitor.classes


def traverse_directory_tree(root_dir):
    json = []           # [{..}, {..}, ... ]

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = f'{dirpath}/{filename}'
            if filename.endswith(".py"):
                fn_list, class_list = source_code_metadata_generator(filepath)

                for function in fn_list:
                    data = {}
                    data['path'] = f"{dirpath}/{filename}"
                    data["function_name"] = f"{function[0]}"
                    data["docstring"] = f"{function[1]}"
                    
                    json.append(data)

                for class_ in class_list:
                    data = {}
                    data['path'] = f"{dirpath}/{filename}"
                    data["class_name"] = f"{class_[0][0]}"
                    data["docstring"] = f"{class_[0][1]}"
                    data["class_methods"] = [method for method in class_[1]]

                    json.append(data)
    
    return json


if __name__ == "__main__":
    res = traverse_directory_tree(sys.argv[1])
    print(res)

"""
for upserting functions/classes JSON format (this will be embedded by the embedding model)


{
    "path": "../..",
    "function/class name": "name",
    "docstring": ".."
    "class_methods": ["method1", "..."]
}
"""


"""
For the metadata

metadata: {
    "path": "../..",
    "function/class name": "name",
    "docstring": ".."
    "methods": [(method1, docstring), (method2, docstring), ...]
}
"""

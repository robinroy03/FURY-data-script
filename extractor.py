import ast
import os
from pprint import pprint as print
import sys

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter
)

rst_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.RST,
    chunk_size=8000,
    chunk_overlap=0
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=8000,
    chunk_overlap=0
)

class Visitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []         # [[fn_name1, docstring, lineno, end_lineno], ......]
        self.classes = []           # [[[class_name1, docstring, lineno, end_lineno], [[method1, docstring, lineno, end_lineno], [method2, docstring, lineno, end_lineno],....]], [...], ...]
        self.depth = 0
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.depth == 1:     # it's inside a class, last elem in self.classes is the specified class
            class_data = self.classes[-1]
            method_data = class_data[1]
            method_data.append([node.name, ast.get_docstring(node), node.lineno, node.end_lineno])
        else:
            self.functions.append([node.name, ast.get_docstring(node), node.lineno, node.end_lineno])

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if self.depth == 1:     # it's inside a class, last elem in self.classes is the specified class
            class_data = self.classes[-1]
            method_data = class_data[1]
            method_data.append([node.name, ast.get_docstring(node), node.lineno, node.end_lineno])
        else:
            self.functions.append([node.name, ast.get_docstring(node), node.lineno, node.end_lineno])
    
    def visit_ClassDef(self, node: ast.ClassDef):
        self.classes.append([[node.name, ast.get_docstring(node), node.lineno, node.end_lineno], []])
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
            if "fury/docs/examples" in filepath and filename.endswith(".py"):
                with open(filepath, "r") as f:
                    source_code = f.read()
                data = {}
                data['type'] = 'documentation_examples'
                data['path'] = f"{dirpath}/{filename}"
                data['content'] = [x.page_content for x in text_splitter.create_documents([source_code])]
                json.append(data)

            elif filename.endswith(".py"):
                fn_list, class_list = source_code_metadata_generator(filepath)

                for function in fn_list:
                    data = {}
                    data['type'] = "function"
                    data['path'] = f"{dirpath}/{filename}"
                    data["function_name"] = f"{function[0]}"
                    data["docstring"] = f"{function[1]}"
                    data["lineno"] = [function[2], function[3]]
                    json.append(data)

                for class_ in class_list:
                    data = {}
                    data['type'] = "class"
                    data['path'] = f"{dirpath}/{filename}"
                    data["class_name"] = f"{class_[0][0]}"
                    data["docstring"] = f"{class_[0][1]}"
                    data["lineno"] = [class_[0][2], class_[0][3]]
                    data["class_methods"] = [method for method in class_[1]]
                    json.append(data)

            elif filename.endswith(".rst"):
                with open(filepath, "r") as f:
                    source_code = f.read()
                data = {}
                data['type'] = 'rst'
                data['path'] = f"{dirpath}/{filename}"
                data['content'] = [x.page_content for x in rst_splitter.create_documents([source_code])]
                json.append(data)

    return json


if __name__ == "__main__":
    res = traverse_directory_tree(sys.argv[1])
    print(res)

"""
for upserting functions/classes JSON format (this will be embedded by the embedding model)
3 types of JSON files inside res

{
    "type": "class",
    "path": "../..",
    class_name": "name",
    "docstring": ".."
    "class_methods": ["method1", "..."]
}

{
    "type": "function",
    "path": "../..",
    "function_name": "name",
    "docstring": ".."
}

{
    "type": "rst",
    "path": "../..",
    "content": [.., .., ..]
}

{
    "type": "documentation_examples",
    "path": "../..",
    "content": ".."
}

"""


"""
For the metadata

{   
    "type": "class",
    "path": "../..",
    "class_name": "name",
    "docstring": "..",
    "lineno": [lineno, end_lineno],
    "class_methods": [(method1, docstring, lineno), (method2, docstring, lineno), ...]
}

{   
    "type: "function",
    "path": "../..",
    "function_name": "name",
    "lineno": [lineno, end_lineno],
    "docstring": ".."
}

{
    "type": "rst",
    "path": "../..",
    "content": [.., .., ..]
}

{
    "type": "documentation_examples",
    "path": "../..",
    "content": ".."
}

"""

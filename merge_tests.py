import os
import ast

def extract_imports_and_functions(filepaths):
    imports = set()
    code_blocks = []
    
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # extract raw source for imports
                start_line = node.lineno - 1
                end_line = node.end_lineno
                imports.add("\n".join(content.splitlines()[start_line:end_line]))
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # extract raw source for functions/classes
                start_line = node.lineno - 1
                end_line = node.end_lineno
                # include decorators
                if node.decorator_list:
                    start_line = node.decorator_list[0].lineno - 1
                code_blocks.append("\n".join(content.splitlines()[start_line:end_line]))
            else:
                # global variables or setup
                start_line = node.lineno - 1
                end_line = node.end_lineno
                code_blocks.append("\n".join(content.splitlines()[start_line:end_line]))
                
    return list(imports), code_blocks

def merge_files(target, sources):
    imports, code_blocks = extract_imports_and_functions(sources)
    with open(target, 'w', encoding='utf-8') as f:
        f.write("\n".join(imports) + "\n\n")
        f.write("\n\n".join(code_blocks) + "\n")
    print(f"Merged {sources} into {target}")

# Define the groupings
mappings = {
    "backend/tests/test_e2e_user_journey.py": [
        "backend/app/tests/test_auth.py",
        "backend/app/tests/test_profile.py",
        "backend/app/tests/test_usage.py",
        "backend/app/tests/test_sessions.py",
        "backend/tests/test_intake.py"
    ],
    "backend/tests/test_engine_logic.py": [
        "backend/tests/test_reasoning.py",
        "backend/tests/test_simulation_pipeline.py",
        "backend/tests/test_simulation_branch_and_queue.py",
        "backend/tests/test_ollama_and_skeleton.py",
        "backend/tests/test_graph_api.py"
    ],
    "backend/tests/test_knowledge_pipeline.py": [
        "backend/tests/test_ingestion.py",
        "backend/tests/test_retrieval.py"
    ],
    "backend/tests/test_evaluations.py": [
        "backend/tests/test_ragas_integration.py",
        "backend/tests/test_retrieval_eval_suite.py"
    ],
    "backend/tests/test_schemas_and_utils.py": [
        "backend/tests/test_schemas.py"
    ]
}

for target, sources in mappings.items():
    merge_files(target, sources)

# Remove the old files to prevent pytest from running duplicates
for sources in mappings.values():
    for source in sources:
        if os.path.exists(source):
            os.remove(source)
            print(f"Removed {source}")

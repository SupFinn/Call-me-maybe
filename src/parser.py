import json
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, ValidationError

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Dict]
    returns: Any

def data_validation(file_path: str = "../data/input/functions_definition.json"
                    ) -> List[FunctionDefinition] | None:
    try:
        with open(Path(file_path), "r") as f:
            data: List[Dict] = json.load(f)
            valid: List[FunctionDefinition] = []
            for function in data:
                try:
                    func = FunctionDefinition(**function)
                    valid.append(func)
                except ValidationError as e:
                    print("Validation failed for function "
                          f"'{function.get('name', 'unknown')}': {e}")
            return valid

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []

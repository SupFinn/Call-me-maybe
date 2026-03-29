import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from llm_sdk.llm_sdk import Small_LLM_Model


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR.parent / "data" / "input"
OUTPUT_DIR = BASE_DIR.parent / "data" / "output"


def load_prompts(
    file_path: Path = INPUT_DIR / "function_calling_tests.json",
) -> List[str]:
    """Load prompts from the input JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
        return []
    except OSError as exc:
        print(f"Error: Could not read {file_path}: {exc}")
        return []

    if not isinstance(data, list):
        print(f"Error: Expected a list in {file_path}")
        return []

    prompts: List[str] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("prompt"), str):
            prompts.append(item["prompt"])
        else:
            print("Warning: Skipping invalid prompt entry")

    return prompts


def load_function_definitions(
    file_path: Path = INPUT_DIR / "functions_definition.json",
) -> Dict[str, Dict[str, Any]]:
    """Load function definitions and return them as a dict keyed by function name."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: Function definitions not found at {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
        return {}
    except OSError as exc:
        print(f"Error: Could not read {file_path}: {exc}")
        return {}

    if not isinstance(data, list):
        print(f"Error: Expected a list in {file_path}")
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            result[item["name"]] = item
        else:
            print("Warning: Skipping invalid function definition entry")

    return result


def load_vocabulary(model: Small_LLM_Model) -> Dict[int, str]:
    """Load the tokenizer vocabulary and build an id -> token mapping."""
    vocab_path = model.get_path_to_vocab_file()

    try:
        with open(vocab_path, "r", encoding="utf-8") as file:
            token_to_id = json.load(file)
    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {vocab_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in vocabulary file {vocab_path}")
        return {}
    except OSError as exc:
        print(f"Error: Could not read vocabulary file {vocab_path}: {exc}")
        return {}

    if not isinstance(token_to_id, dict):
        print("Error: Vocabulary file does not contain a dictionary")
        return {}

    id_to_token: Dict[int, str] = {}
    for token, token_id in token_to_id.items():
        if isinstance(token, str) and isinstance(token_id, int):
            id_to_token[token_id] = token

    return id_to_token


def constrained_decode(
    model: Small_LLM_Model,
    prompt_input_ids: List[int],
    id_to_token: Dict[int, str],
    max_tokens: int = 100,
) -> str:
    """
    Generate only the continuation after the prompt.

    This is a basic JSON-oriented constrained decoder baseline.
    It fixes the current crash and adds simple structure constraints,
    but it does NOT yet fully enforce the exact function schema.
    """
    generated_ids = prompt_input_ids.copy()
    prompt_length = len(prompt_input_ids)

    started_json = False
    open_braces = 0
    inside_string = False

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(generated_ids)

        if not isinstance(logits, list) or not logits:
            print("Warning: Model returned empty logits")
            break

        # SDK returns a flat list[float] for the NEXT token.
        next_token_logits = np.array(logits, dtype=float)

        for token_id, token_str in id_to_token.items():
            if token_id >= len(next_token_logits):
                continue

            # Force the very first generated token to be "{"
            if not started_json:
                if token_str != "{":
                    next_token_logits[token_id] = -np.inf
                continue

            # Very basic structural restrictions
            if inside_string and token_str in ["{", "}", ":", ","]:
                next_token_logits[token_id] = -np.inf

        if np.all(np.isneginf(next_token_logits)):
            print("Warning: No valid token available during decoding")
            break

        next_token_id = int(np.argmax(next_token_logits))
        generated_ids.append(next_token_id)

        token_str = id_to_token.get(next_token_id, "")

        if not started_json:
            if token_str == "{":
                started_json = True
                open_braces = 1
            continue

        if token_str == '"':
            inside_string = not inside_string
        elif not inside_string:
            if token_str == "{":
                open_braces += 1
            elif token_str == "}":
                open_braces -= 1
                if open_braces == 0:
                    break

    generated_only_ids = generated_ids[prompt_length:]
    return model.decode(generated_only_ids)


def build_fallback_result(prompt: str) -> Dict[str, Any]:
    """Build a safe fallback result."""
    return {
        "prompt": prompt,
        "name": "unknown_function",
        "parameters": {},
    }


def call_functions_from_prompts() -> None:
    """Main entry point."""
    prompts = load_prompts()
    function_definitions = load_function_definitions()
    model = Small_LLM_Model()
    id_to_token = load_vocabulary(model)

    if not prompts:
        print("Error: No prompts loaded")
        return

    if not function_definitions:
        print("Warning: No function definitions loaded")

    if not id_to_token:
        print("Error: Vocabulary could not be loaded")
        return

    results: List[Dict[str, Any]] = []

    for prompt in prompts:
        try:
            encoded = model.encode(prompt)
            input_ids = encoded.squeeze(0).tolist()

            generated_json_str = constrained_decode(
                model=model,
                prompt_input_ids=input_ids,
                id_to_token=id_to_token,
                max_tokens=100,
            )

            try:
                parsed = json.loads(generated_json_str)

                if (
                    isinstance(parsed, dict)
                    and isinstance(parsed.get("name"), str)
                    and isinstance(parsed.get("parameters"), dict)
                ):
                    result = {
                        "prompt": prompt,
                        "name": parsed["name"],
                        "parameters": parsed["parameters"],
                    }
                else:
                    print(f"Warning: Invalid JSON schema for prompt: {prompt}")
                    print(f"Generated continuation: {generated_json_str!r}")
                    result = build_fallback_result(prompt)

            except json.JSONDecodeError:
                print(f"Warning: Generated text is not valid JSON for prompt: {prompt}")
                print(f"Generated continuation: {generated_json_str!r}")
                result = build_fallback_result(prompt)

        except Exception as exc:
            print(f"Error while processing prompt {prompt!r}: {exc}")
            result = build_fallback_result(prompt)

        results.append(result)

    output_path = OUTPUT_DIR / "function_calling_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"Error: Could not write output file {output_path}: {exc}")
        return

    print(f"Generation complete! Results saved to {output_path}")


if __name__ == "__main__":
    call_functions_from_prompts()
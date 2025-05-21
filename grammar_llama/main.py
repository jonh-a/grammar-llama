from pynput.keyboard import Controller, Key, GlobalHotKeys
from ollama import chat, ps, show, ResponseError
from pydantic import BaseModel
from pyperclip import copy, paste
from httpx import ConnectError
from difflib import unified_diff
from typing import List, Union, Literal
import time
import os
import re
import sys


DEFAULT_MODEL = "gemma3"
DEFAULT_PROMPT = (
    "Correct the spelling, grammar, or phrasing issues in the following text. "
    "Try to match the tone of the original message. "
    "The response will be a JSON object that contains:\n "
    " - original_grammar_strength: A ranking (1-5, 1 being poor/nearly incomprehensible, and 5 not requiring any changes)\n"
    " - corrected_text: The corrected text\n"
    " - summary_of_corrections: A brief summary of the changes made\n"
    " - tone: One word describing the tone of the message (friendly, casual, professional, sarcastic, etc.)"
    "Use only JSON-safe characters in your response."
)

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

def get_model() -> str:
    model = os.getenv("CHECKER_MODEL")
    return model if model else DEFAULT_MODEL


def get_prompt() -> str:
    prompt = os.getenv("CHECKER_PROMPT")
    return prompt if prompt else DEFAULT_PROMPT

def is_mac() -> bool:
    return sys.platform.startswith('darwin')


MODEL = get_model()
PROMPT = get_prompt()
IS_MAC = is_mac()

controller = Controller()


class Response(BaseModel):
    original_grammar_strength: Literal["1", "2", "3", "4", "5"]
    corrected_text: str
    summary_of_corrections: str
    tone: str


def get_hotkey_combo() -> str:
    hotkey_specification = os.getenv('CHECKER_HOTKEY')

    if hotkey_specification:
        return hotkey_specification
    elif sys.platform.startswith('darwin'):
        return '<ctrl>+<cmd>+a'
    else:
        return '<ctrl>+<alt>+a'


def run_startup_tasks() -> None:
    try:
        ps()
        show(MODEL)
    except ConnectError:
        print(" - Failed to connect to Ollama.")
        sys.exit(1)
    except ResponseError:
        print(f" - Ollama model {MODEL} not found.")
        sys.exit(1)

    print(" + Startup tasks passed.")
    print(f" + Using model: {MODEL}")
    print(f" + Using prompt: {PROMPT}\n")


def chunk_text(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def print_diff(original_text: str, corrected_text: str) -> None:
    original_chunks = chunk_text(original_text)
    corrected_chunks = chunk_text(corrected_text)
    diff = unified_diff(original_chunks, corrected_chunks, lineterm='', fromfile='original', tofile='corrected')

    for line in diff:
        if line.startswith('+') and not line.startswith('+++'):
            print(f"{GREEN}{line}{RESET}")
        elif line.startswith('-') and not line.startswith('---'):
            print(f"{RED}{line}{RESET}")
        else:
            print(line)


def copy_text_at_cursor() -> None:
    key = Key.cmd if is_mac() else Key.ctrl
    with controller.pressed(key):
        controller.press('c')
        controller.release('c')
    time.sleep(0.1)


def correct_grammar(text: str) -> Union[Response, None]:
    print(f" + Awaiting response from LLM...")
    try:
        response = chat(
            model=MODEL, 
            messages=[
                {
                    "role": "user",
                    "content": PROMPT,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            format=Response.model_json_schema(),
        )

        if response and response.message and response.message.content:
            response_obj = Response.model_validate_json(response.message.content)

            print(f" + Received corrected content: \n{response_obj.corrected_text}\n")
            return response_obj
    
        else:
            print(" - Failed to receive response from LLM.")
        
        return None
    except ResponseError:
        print(" - Unable to get response from Ollama.")
    except Exception as e:
        print(f" - An unexpected error occurred: {e}")


def paste_text_at_cursor() -> None:
    key = Key.cmd if is_mac() else Key.ctrl
    with controller.pressed(key):
        controller.press('v')
        controller.release('v')
    time.sleep(0.1)


def summarize_grammar(response: Response) -> None:
    print(f"\n + Original text score: {response.original_grammar_strength}")
    print(f"\n + Original text tone: {response.tone}")
    print(f" + Summary of corrections: {response.summary_of_corrections}\n")


def on_activate() -> None:
    copy_text_at_cursor()
    original_text = paste()

    print(f"\n + Copied text:\n{original_text}\n")
    
    response = correct_grammar(original_text)

    if response and response.corrected_text:
        print_diff(original_text, response.corrected_text)
        summarize_grammar(response)
        copy(response.corrected_text)
        paste_text_at_cursor()
    else:
        print(" - Correction failed; skipping paste.")


def main():
    try:
        run_startup_tasks()

        with GlobalHotKeys({get_hotkey_combo(): on_activate}) as h:
            h.join()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(" - An unhandled exception occurred: {e}")


if __name__ == "__main__":
    main()
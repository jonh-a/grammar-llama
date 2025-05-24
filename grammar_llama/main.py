from pynput.keyboard import Controller, Key, GlobalHotKeys
from ollama import ps, show, ResponseError, AsyncClient
from pydantic import BaseModel
from pyperclip import copy, paste  # type: ignore
from httpx import ConnectError
from difflib import unified_diff
from typing import List, Union, Literal, Optional
import asyncio
import os
import re
import sys


DEFAULT_MODEL = "gemma3"
DEFAULT_PROMPT = (
    "Correct the spelling, grammar, or phrasing issues in the following text. "
    "Try to match the tone of the original message. "
    "The response will be a JSON object that contains:\n "
    " - original_grammar_strength: A ranking (1-3, 1 if incomprehensible, 2 if moderate changes needed, 3 if little to no changes needed)\n"
    " - corrected_text: The corrected text\n"
    " - summary_of_corrections: A brief summary of the changes made\n"
    " - tone: One word describing the tone of the message (friendly, casual, professional, sarcastic, etc.)"
    "Use only JSON-safe characters in your response."
)

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


def get_model() -> str:
    model = os.getenv("CHECKER_MODEL")
    return model if model else DEFAULT_MODEL


def get_prompt() -> str:
    prompt = os.getenv("CHECKER_PROMPT")
    return prompt if prompt else DEFAULT_PROMPT


def is_mac() -> bool:
    return sys.platform.startswith("darwin")


def chunk_text(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


MODEL = get_model()
PROMPT = get_prompt()
IS_MAC = is_mac()


class Response(BaseModel):
    original_grammar_strength: Literal[1, 2, 3]
    corrected_text: str
    summary_of_corrections: str
    tone: str


class GrammarChecker:
    def __init__(self):
        self.controller = Controller()
        self.model = get_model()
        self.prompt = get_prompt()
        self.is_mac = is_mac()
        self.modifier_key = Key.cmd if self.is_mac else Key.ctrl
        self.hotkey = self.get_hotkey_combo()

        self.current_task: Optional[asyncio.Task] = None
        self.client = AsyncClient()
        self.lock = asyncio.Lock()
        self.cancelled = False

        try:
            self.run_startup_tasks()
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            print(" - An unhandled exception occurred: {e}")

    def get_hotkey_combo(self) -> str:
        hotkey_specification = os.getenv("CHECKER_HOTKEY")

        if hotkey_specification:
            return hotkey_specification
        elif self.is_mac:
            return "<ctrl>+<cmd>+a"
        else:
            return "<ctrl>+<alt>+a"

    def run_startup_tasks(self) -> None:
        try:
            ps()
            show(self.model)
        except ConnectError:
            print(" - Failed to connect to Ollama.")
            sys.exit(1)
        except ResponseError:
            print(f" - Ollama model {self.model} not found.")
            sys.exit(1)

        print(" + Startup tasks passed.")
        print(f" + Using model: {self.model}")
        print(f" + Using prompt: {self.prompt}\n")

    def print_diff(self, original_text: str, corrected_text: str) -> None:
        original_chunks = chunk_text(original_text)
        corrected_chunks = chunk_text(corrected_text)
        diff = unified_diff(
            original_chunks,
            corrected_chunks,
            lineterm="",
            fromfile="original",
            tofile="corrected",
        )

        for line in diff:
            if line.startswith("+") and not line.startswith("+++"):
                print(f"{GREEN}{line}{RESET}")
            elif line.startswith("-") and not line.startswith("---"):
                print(f"{RED}{line}{RESET}")
            else:
                print(line)

    async def copy_text_at_cursor(self) -> str:
        with self.controller.pressed(self.modifier_key):
            self.controller.press("c")
            self.controller.release("c")
        await asyncio.sleep(0.1)
        return paste()

    async def correct_grammar(self, text: str) -> Union[Response, None]:
        print(f" + Awaiting response from LLM...")
        try:
            chat_task = asyncio.create_task(
                self.client.chat(
                    model=self.model,
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
            )

            while not chat_task.done():
                if self.cancelled:
                    chat_task.cancel()
                    raise asyncio.CancelledError()
                await asyncio.sleep(0.1)

            response = await chat_task

            if response and response.message and response.message.content:
                response_obj = Response.model_validate_json(response.message.content)

                print(
                    f" + Received corrected content: \n{response_obj.corrected_text}\n"
                )
                return response_obj

            else:
                print(" - Failed to receive response from LLM.")

            return None
        except asyncio.CancelledError:
            print(" + Cancelling LLM request...")
            raise
        except ResponseError:
            print(" - Unable to get response from Ollama.")
            return None
        except Exception as e:
            print(f" - An unexpected error occurred: {e}")
            return None

    async def paste_text_at_cursor(self) -> None:
        with self.controller.pressed(self.modifier_key):
            self.controller.press("v")
            self.controller.release("v")
        await asyncio.sleep(0.1)

    def summarize_grammar(self, response: Response) -> None:
        print(f"\n + Original text score: {response.original_grammar_strength}")
        print(f"\n + Original text tone: {response.tone}")
        print(f" + Summary of corrections: {response.summary_of_corrections}\n")

    async def process_text(self) -> None:
        self.cancelled = False
        async with self.lock:
            original_text = await self.copy_text_at_cursor()
            print(f"\n + Copied text:\n{original_text}\n")

            response = await self.correct_grammar(original_text)

            if response and response.corrected_text:
                self.print_diff(original_text, response.corrected_text)
                self.summarize_grammar(response)
                copy(response.corrected_text)
                await self.paste_text_at_cursor()
            else:
                print(" - Correction failed; skipping paste.")

    async def handle_hotkey(self) -> None:
        if self.current_task and not self.current_task.done():
            self.cancelled = True
            self.current_task.cancel()

        self.cancelled = False
        self.current_task = asyncio.create_task(self.process_text())


async def main_async():
    checker = GrammarChecker()
    loop = asyncio.get_event_loop()

    def on_activate():
        asyncio.run_coroutine_threadsafe(coro=checker.handle_hotkey(), loop=loop)

    hotkey_handler = GlobalHotKeys({checker.hotkey: on_activate})
    hotkey_handler.start()

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        hotkey_handler.stop()
        if checker.current_task and not checker.current_task.done():
            checker.current_task.cancel()
            try:
                loop.run_until_complete(checker.current_task)
            except asyncio.CancelledError:
                pass


def main():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(main_async())
        except KeyboardInterrupt:
            print(" + Shutting down...")
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()

            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
    except Exception as e:
        print(f" - An unhandled exception occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

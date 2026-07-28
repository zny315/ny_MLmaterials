import argparse
import atexit
import base64
import hashlib
import html
import json
import os
import re
import signal
import shutil
import sys
import threading
import time
import traceback
import unicodedata
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from public_utils import ascii_text, console_print as print, install_ascii_exception_hook
    from schema import (
        ENTITY_COLUMNS,
        IDENTITY_COLUMNS,
        MERGED_COLUMNS,
        RELEASE_MERGED_COLUMNS,
        PART_COLUMNS,
        PERFORMANCE_COLUMNS,
        PERFORMANCE_IDENTITY_COLUMNS,
        SOURCE_COLUMNS,
        canonicalize_dataframe_columns,
    )
except ModuleNotFoundError:
    from pipeline.public_utils import ascii_text, console_print as print, install_ascii_exception_hook
    from pipeline.schema import (
        ENTITY_COLUMNS,
        IDENTITY_COLUMNS,
        MERGED_COLUMNS,
        RELEASE_MERGED_COLUMNS,
        PART_COLUMNS,
        PERFORMANCE_COLUMNS,
        PERFORMANCE_IDENTITY_COLUMNS,
        SOURCE_COLUMNS,
        canonicalize_dataframe_columns,
    )

install_ascii_exception_hook()


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "inputs" / "mineru"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "pipeline"
DEFAULT_API_KEYS_FILE = Path(os.getenv("OPENAI_API_KEYS_FILE", REPO_ROOT / ".secrets" / "api_keys.txt"))
DEFAULT_API_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")


def read_excel_canonical(path, *args, **kwargs) -> pd.DataFrame:
    dataframe = pd.read_excel(path, *args, **kwargs)
    return canonicalize_dataframe_columns(dataframe, context=str(path))


class EvaPipelineBaseAnalyzer:
    _hooks_registered = False

    def __init__(self, output_excel: str, model_name: str, input_folder: str):
        self.excel_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self._api_key_queue = None

        self.input_folder = input_folder
        self.output_excel = output_excel
        self.output_dir = os.path.dirname(os.path.abspath(self.output_excel))
        os.makedirs(os.path.join(self.output_dir, "api_raw"), exist_ok=True)

        self.api_url = DEFAULT_API_URL
        self.api_key = os.getenv("OPENAI_API_KEY") or ""
        self.model_name = model_name
        model_key = str(self.model_name).lower()
        self.supports_vision = any(
            key in model_key
            for key in ["gpt-4o", "gpt-5", "vision", "multimodal", "gemini", "vl", "glm", "claude", "qwen3.6"]
        )
        self.target_image_categories = {"scheme", "structure", "figure", "chart", "image", "table"}

        self.session = requests.Session()
        retries = Retry(total=1, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

        self.running = True
        self.dry_run = False
        self.processed_files = set()
        self.run_start_time = datetime.now()
        self.run_processed = 0
        self.run_success = 0
        self.run_failed = 0
        self.run_empty = 0
        self._has_summarized = False
        self.column_order = []
        self.prompt_template = ""
        self.temperature = 0.1

        self.log_file_path = os.path.join(self.output_dir, "run.log")
        log_mode = "a" if os.path.exists(self.log_file_path) else "w"
        with open(self.log_file_path, log_mode, encoding="utf-8") as f:
            f.write(ascii_text(f"=== Run started at {datetime.now()} ===\n"))

        self.log(f"Model '{self.model_name}' initialized. Vision enabled: {'yes' if self.supports_vision else 'no'}")
        self.log("API endpoint configured.")
        self.log(f"Output directory: {self.output_dir}")
        self.log(f"Run log: {self.log_file_path}")

        if not self.__class__._hooks_registered:
            atexit.register(self._safe_exit_summary)
            try:
                signal.signal(signal.SIGINT, self._handle_interrupt)
            except Exception:
                pass
            self.__class__._hooks_registered = True

    def log(self, message: str):
        safe_message = ascii_text(message)
        print(safe_message)
        with self.log_lock:
            try:
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] {safe_message}\n")
            except Exception:
                pass

    def read_mineru_output(self, paper_dir: str) -> tuple[str, List[str]]:
        markdown_text = ""
        image_paths: List[str] = []
        paper_dir = os.path.normpath(os.path.abspath(paper_dir))

        target_dir = paper_dir
        auto_dir = os.path.normpath(os.path.join(paper_dir, "auto"))
        if os.path.exists(auto_dir) and os.path.isdir(auto_dir):
            target_dir = auto_dir
            self.log(f"Detected MinerU auto directory; using: {os.path.basename(auto_dir)}")

        try:
            md_files = sorted(f for f in os.listdir(target_dir) if f.lower().endswith((".md", ".txt")))
            if md_files:
                md_path = os.path.normpath(os.path.join(target_dir, md_files[0]))
                with open(md_path, "r", encoding="utf-8", errors="ignore") as f:
                    markdown_text = f.read()
                self.log(f"Read Markdown file: {os.path.basename(md_path)}")
        except Exception as exc:
            self.log(f"Failed to read Markdown file: {exc}")

        images_dir = os.path.normpath(os.path.join(target_dir, "images"))
        if os.path.exists(images_dir):
            try:
                for file_name in sorted(os.listdir(images_dir)):
                    if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff")):
                        img_path = os.path.normpath(os.path.join(images_dir, file_name))
                        if len(img_path) <= 250:
                            image_paths.append(img_path)
                        else:
                            self.log(f"Skipped image with an overlong path: {os.path.basename(file_name)}")
                self.log(f"Loaded {len(image_paths)} images")
            except Exception as exc:
                self.log(f"Failed to read image directory: {exc}")

        return markdown_text, image_paths

    def preprocess_text(self, text):
        if not text:
            return ""
        if re.search(r"\[!!!\s+[^]\n]{4,80}\s+!!!\]", text):
            return text
        if len(text) > 100 and "\n" not in text:
            return text

        text = re.sub(r"([a-zA-Z@\(\)\[\]])\s*[\n\s]+(\d+)\b", r"\1\2", text)
        text = re.sub('(\\d+)\\s*\\n\\s*(\xb1)\\s*\\n\\s*(\\d+)', r"\1 \2 \3", text)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        merged_lines = []
        for line in lines:
            if len(re.findall(r"\d+\.\d*", line)) >= 6 and "\u00b1" in line:
                line = f"[DATA_SOUP_IGNORED] {line}"
            if line.count("\u00b1") + line.count("+-") > 5:
                line = f"[STRUCTURE_LOST_DATA] {line}"
            if not merged_lines:
                merged_lines.append(line)
                continue
            prev = merged_lines[-1]
            if not re.search('[\\.\\\\!\uff1a:;]$', prev) and not re.match(r"^[A-Z]", line):
                merged_lines[-1] += " " + line
            else:
                merged_lines.append(line)
        return "\n\n".join(merged_lines)

    def _save_api_preview(self, messages, file_id):
        preview_dir = os.path.join(self.output_dir, "api_preview")
        os.makedirs(preview_dir, exist_ok=True)
        safe_file_id = os.path.splitext(file_id)[0]
        preview_path = os.path.join(preview_dir, f"preview_{safe_file_id}.html")

        chunks = [
            "<html><body style='font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; background: #f4f4f4;'>",
            "<div style='max-width: 880px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px;'>",
            f"<h1>API Request Preview: {html.escape(file_id)}</h1>",
        ]
        for msg in messages:
            chunks.append("<div style='margin-bottom: 20px; padding: 16px; background: #f9f9f9; border-left: 4px solid #007bff;'>")
            chunks.append(f"<b>Role: {html.escape(str(msg.get('role', '')))}</b><hr>")
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        chunks.append(f"<pre style='white-space: pre-wrap'>{html.escape(self.preprocess_text(item.get('text', '')))}</pre>")
                    elif item.get("type") == "image_url":
                        img_data = item.get("image_url", {}).get("url", "")
                        chunks.append(f"<img src='{html.escape(img_data)}' style='max-width:100%; border:1px solid #ddd; margin:12px 0;'>")
            else:
                chunks.append(f"<pre style='white-space: pre-wrap'>{html.escape(self.preprocess_text(str(content or '')))}</pre>")
            chunks.append("</div>")
        chunks.append(f"<p style='color:#666;font-size:12px'>Generated {datetime.now():%Y-%m-%d %H:%M:%S}</p>")
        chunks.append("</div></body></html>")

        with open(preview_path, "w", encoding="utf-8") as f:
            f.write("\n".join(chunks))
        self.log(f"[Preview] Generated: api_preview/preview_{safe_file_id}.html")

    def _extract_chat_content(self, response_json: Dict) -> str:
        choice = (response_json.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content or "")

    def _extract_stream_content(self, response) -> tuple:
        parts = []
        saw_done = False
        finish_reason = None
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                saw_done = True
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = choice.get("finish_reason")
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            content = delta.get("content", message.get("content", ""))
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        parts.append(str(item.get("text") or item.get("content") or ""))
                    else:
                        parts.append(str(item))
        return "".join(parts), saw_done, finish_reason

    def call_llm_api(self, content_blocks: List[Dict], file_id="unknown"):
        if self.supports_vision:
            multi_content = [{"type": "text", "text": self.prompt_template}]
            for block in content_blocks:
                if block["type"] == "text" and block.get("content"):
                    multi_content.append({"type": "text", "text": self.preprocess_text(block["content"])})
                elif block["type"] == "image" and block.get("content") and block.get("category") in self.target_image_categories:
                    image_id = block.get("image_id", "Unknown")
                    multi_content.append({"type": "text", "text": f"\n[System Image ID: {image_id}]"})
                    multi_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{block['content']}"},
                    })
            messages = [{"role": "user", "content": multi_content}]
        else:
            full_text = self.prompt_template
            last_page = -1
            for block in content_blocks:
                if block["page"] != last_page:
                    full_text += f"\n\n--- Page {block['page']} ---\n"
                    last_page = block["page"]
                if block["type"] == "text" and block.get("content"):
                    full_text += self.preprocess_text(block["content"]) + "\n"
            messages = [{"role": "user", "content": full_text}]

        self._save_api_preview(messages, file_id)
        if self.dry_run:
            self.log("[Dry Run] Generated the API request preview and skipped the API call.")
            return None

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": int(getattr(self, "max_tokens", 65536) or 65536),
            "temperature": float(getattr(self, "temperature", 0.1)),
            "response_format": {"type": "json_object"},
        }
        if getattr(self, "stream", False):
            payload["stream"] = True
        attempts = max(1, int(getattr(self, "api_retries", 2) or 2))
        connect_timeout = float(getattr(self, "connect_timeout", 10) or 10)
        read_timeout = float(getattr(self, "read_timeout", 540) or 540)
        retry_base_delay = float(getattr(self, "retry_base_delay", 15) or 15)
        rate_limit_sleep = float(getattr(self, "rate_limit_sleep", 1800) or 1800)
        rate_limit_retries = max(0, int(getattr(self, "rate_limit_retries", 3) or 3))
        key_queue = getattr(self, "_api_key_queue", None)
        api_key = key_queue.get() if key_queue is not None else self.api_key

        def is_rate_limited(status_code=None, text="") -> bool:
            lowered = str(text or "").lower()
            if status_code == 429:
                return True
            keywords = [
                "rate limit", "ratelimit", "too many requests", "quota",
                "requests per minute", "rpm", "tpm", "throttl",
            ]
            return any(keyword in lowered for keyword in keywords)

        try:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            attempt = 0
            rate_limit_count = 0
            while attempt < attempts and self.running:
                rate_limited = False
                try:
                    response = self.session.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=(connect_timeout, read_timeout),
                        stream=bool(getattr(self, "stream", False)),
                    )
                    if response.status_code == 200:
                        if getattr(self, "stream", False):
                            content, saw_done, finish_reason = self._extract_stream_content(response)
                            res_json = {}
                            if not saw_done:
                                self.log(f"Streaming response did not include [DONE] (finish_reason={finish_reason or '#'}, attempt={attempt + 1}/{attempts})")
                                content = ""
                            elif str(finish_reason or "").lower() in {"length", "max_tokens", "max_output_tokens"}:
                                self.log(f"Streaming response was truncated (finish_reason={finish_reason}, attempt={attempt + 1}/{attempts})")
                                content = ""
                            elif content:
                                try:
                                    self._smart_fix_json(content)
                                except (ValueError, json.JSONDecodeError):
                                    self.log(f"Streaming response was not complete JSON (finish_reason={finish_reason or '#'}, attempt={attempt + 1}/{attempts})")
                                    content = ""
                        else:
                            res_json = response.json()
                            content = self._extract_chat_content(res_json)
                            finish_reason = (res_json.get("choices") or [{}])[0].get("finish_reason")
                            if str(finish_reason or "").lower() in {"length", "max_tokens", "max_output_tokens"}:
                                self.log(f"API response was truncated (finish_reason={finish_reason}, attempt={attempt + 1}/{attempts})")
                                content = ""
                            elif content:
                                try:
                                    self._smart_fix_json(content)
                                except (ValueError, json.JSONDecodeError) as exc:
                                    self.log(f"API response was not complete JSON: {exc} (finish_reason={finish_reason or '#'}, attempt={attempt + 1}/{attempts})")
                                    content = ""
                        if content:
                            return content
                        finish_reason = finish_reason or (res_json.get("choices") or [{}])[0].get("finish_reason", "#")
                        usage = res_json.get("usage", {})
                        self.log(f"API returned empty content (finish_reason={finish_reason}, usage={usage}, attempt={attempt + 1}/{attempts})")
                    else:
                        rate_limited = is_rate_limited(response.status_code, response.text)
                        self.log(f"API error: {response.status_code} {response.text[:150]} (attempt={attempt + 1}/{attempts})")
                except requests.exceptions.RequestException as exc:
                    rate_limited = is_rate_limited(text=str(exc))
                    self.log(f"Network error: {exc} (attempt={attempt + 1}/{attempts})")

                if rate_limited and rate_limit_count < rate_limit_retries:
                    rate_limit_count += 1
                    self.log(f"Rate limit reached; waiting {rate_limit_sleep / 60:.0f} minutes before retry ({rate_limit_count}/{rate_limit_retries})")
                    time.sleep(rate_limit_sleep)
                    continue

                if attempt < attempts - 1:
                    delay = retry_base_delay * (2 ** attempt)
                    self.log(f"Retrying in {delay:.0f} seconds")
                    time.sleep(delay)
                attempt += 1
            return None
        finally:
            if key_queue is not None and api_key:
                key_queue.put(api_key)

    def _smart_fix_json(self, raw_text: str) -> str:
        if not raw_text or not raw_text.strip():
            raise ValueError("empty JSON response")
        text = raw_text.strip()
        if text.startswith("```"):
            fenced = re.fullmatch(r"```(?:json)?\s*(.*)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
            if not fenced:
                raise ValueError("incomplete JSON code fence")
            text = fenced.group(1).strip()
        if not text.startswith("{") or not text.endswith("}"):
            raise ValueError("JSON response must contain exactly one complete object")

        text = re.sub(r",\s*([\}\]])", r"\1", text)
        text = self._repair_common_model_json_errors(text)
        if not self._json_structure_is_balanced(text):
            raise ValueError("unbalanced or truncated JSON structure")
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("JSON response root must be an object")
        return text

    @staticmethod
    def _json_structure_is_balanced(text: str) -> bool:
        stack = []
        in_string = False
        escaped = False
        pairs = {"}": "{", "]": "["}
        for char in text:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "[{":
                stack.append(char)
            elif char in "]}":
                if not stack or stack.pop() != pairs[char]:
                    return False
        return not stack and not in_string and not escaped

    def _repair_common_model_json_errors(self, text: str) -> str:
        """Fix recurring complete-response JSON glitches without guessing missing content."""
        # Some providers/models occasionally add one extra object close between samples:
        # ... "parts": [] }] } }, { "sample_name": ...
        text = re.sub(
            r"\}\]\}\},\s*(\{\s*\"sample_name\")",
            r"}]},\1",
            text,
        )
        return text

    def _raw_has_rows_in_excel(self, file_id: str) -> bool:
        if not os.path.exists(self.output_excel):
            return False
        try:
            df = read_excel_canonical(self.output_excel, usecols=[5])
        except Exception:
            return False
        if df.empty:
            return False
        return df.iloc[:, 0].astype(str).eq(str(file_id)).any()

    def _drop_rows_for_file(self, file_id: str) -> int:
        if not os.path.exists(self.output_excel):
            return 0
        try:
            df = read_excel_canonical(self.output_excel)
        except Exception:
            return 0
        if "source_id" not in df.columns:
            return 0

        mask = df["source_id"].astype(str).eq(str(file_id))
        count = int(mask.sum())
        if not count:
            return 0

        df = df.loc[~mask]
        df.to_excel(self.output_excel, index=False)
        return count

    def _force_remove_existing_raw(self, raw_path: str, file_id: str) -> None:
        with self.excel_lock:
            dropped_rows = self._drop_rows_for_file(file_id)
        if os.path.exists(raw_path):
            os.remove(raw_path)
        self.log(f"[{self.task_name}] force rerun: removed existing raw and {dropped_rows} old Excel rows for {file_id}")

    def _move_bad_raw(self, raw_path: str, reason: str) -> str:
        bad_dir = os.path.join(self.output_dir, "api_raw_bad")
        os.makedirs(bad_dir, exist_ok=True)
        stem, ext = os.path.splitext(os.path.basename(raw_path))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_reason = re.sub(r"[^A-Za-z0-9_.-]+", "_", reason).strip("_")[:40] or "bad"
        bad_path = os.path.join(bad_dir, f"{stem}.{safe_reason}.{timestamp}{ext}")
        shutil.move(raw_path, bad_path)
        return bad_path

    def _raw_is_complete_json(self, raw_path: str) -> bool:
        try:
            with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read().strip()
        except Exception:
            return False

        try:
            self._smart_fix_json(text)
            return True
        except (ValueError, json.JSONDecodeError):
            return False

    def _rows_from_existing_raw(self, raw_path: str, file_id: str, paper_dir: str, article_idx: int) -> List[Dict]:
        with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
            response_text = f.read()
        self._smart_fix_json(response_text)
        rows = self.parse_task_response(response_text, article_idx)
        return self.attach_source_metadata(rows, file_id, paper_dir)

    def append_to_excel(self, rows):
        append_rows_to_excel(self.output_excel, rows, self.column_order)

    def summarize_run(self):
        if self._has_summarized:
            return
        self._has_summarized = True
        duration = (datetime.now() - self.run_start_time).total_seconds()
        self.log("\n" + "=" * 20 + " Run Summary " + "=" * 20)
        self.log(f"Duration: {duration:.2f} seconds")
        self.log(f"Files processed: {self.run_processed}")
        self.log(f"  - Successful: {self.run_success}")
        self.log(f"  - Valid empty results: {self.run_empty}")
        self.log(f"  - Failed: {self.run_failed}")
        self.log("=" * 50)

    def _safe_exit_summary(self):
        if not self._has_summarized:
            self.summarize_run()

    def _handle_interrupt(self, signum, frame):
        self.log("\nInterrupt received; terminating immediately.")
        self.running = False
        os._exit(130)


ENTITY_FORMULATION_PROMPT = """
You are a materials researcher building a standardized EVA/PE/POE flame-retardant formulation database.

Extract ALL sample formulation data from the paper, including text, tables, captions, schemes, and images.
This task is FORMULATION ONLY. Do NOT extract performance metrics here.
Return valid JSON only. Do not output Markdown, explanations, or extra text.
Use "#" for missing values. Do NOT guess values that are not explicitly supported by the paper.

[ABSOLUTE FORMULATION RECALL MANDATE]
Your first priority is complete formulation coverage. A formulation table is a sample roster, not an example list.
Before writing JSON, mentally build a checklist of every sample row/column label in all formulation tables, experimental formulation paragraphs, captions, and sample-code definitions. The final JSON samples array must cover that checklist.
Do NOT stop after extracting neat matrix, control samples, the best-performing sample, the first two rows, or representative examples.
Do NOT collapse a series into one sample. Gradient samples such as EVM/40IFR, EVM/39IFR/1BAC, EVM/38IFR/2BAC, EVM/37IFR/3BAC, EVM/36IFR/4BAC, EVM/25IFR, and EVM/23.1IFR/1.9BAC are separate samples and must each be output.
Do NOT omit samples merely because their formulation is encoded in the sample name. If the paper defines the naming rule, decode each sample name into structured entities and loading values.
If a table gives many columns as samples and rows as components, transpose the table logically: each sample column becomes one sample object, and each nonzero component row becomes an entity for that sample.
If a table gives many rows as samples and columns as components, every sample row must become one sample object.
If a paper has multiple formulation tables or formulation blocks, process every block. Keep parent context in sample_name or formulation_notes when the same short label appears in different blocks.
After extracting, verify internally: number of JSON samples should be at least the number of distinct experimental formulation samples visible in the formulation table(s), unless the table is not an EVA/PE/POE-family experimental formulation table.

[CORE DESIGN PRINCIPLE]
This task outputs two linked tables:
1. entity table: one row per actual formulation entity added into the EVA/PE/POE system.
2. entity parts table: one row per chemically meaningful part inside a modified, grafted, coated, encapsulated, hybrid, or core-shell entity.

The entity is the material that is actually added to the polymer formulation.
For KH550-treated MDH, DOPO-g-APP, APP@SiO2, microencapsulated APP, MH@Fe-GA, or similar materials, the entity is the FINAL treated/grafted/coated product, not the unmodified raw material alone.
The raw material, shell, grafted molecule, ligand, coupling agent, dopant, or coating material must be recorded as parts of that entity.

Do NOT let a modifier, coupling agent, shell, coating layer, or grafted molecule consume the second flame-retardant slot later unless it is independently added to the formulation as its own flame-retardant entity.

[CRITICAL SAMPLE COVERAGE RULES]
1. Extract every formulation sample mentioned anywhere in the paper, including samples only appearing in tables or captions.
2. If a sample appears in multiple places, merge its formulation information into one sample object.
3. If a sample exists but its exact formulation is not provided, keep the sample and output "#" for missing fields.
4. Do NOT copy formulations from adjacent samples. Do NOT forward-fill loading values.
5. Preserve original sample names exactly as written, including hyphens, slashes, ratios, and suffixes.
6. Formulation tables often contain the only reliable sample roster. You must extract all rows/columns from tables titled or described as formulation, composition, recipe, sample codes, materials, compounds, formulations, or preparation.
7. Footnotes and surrounding text such as "IFR is APP/DPER = 3/1", "odd samples contain 90 phr MH", "even samples contain 150 phr MH", "A series is epoxy-silane treated", or "B series is stearic-acid treated" are binding definitions. Apply them to every affected sample, not only to the nearby row.
8. Sample-code families like A1-A8, B1-B8, VE1/VE2/VE3, EVA1-EVA4, IFR/BAC gradients, MH/ZB gradients, OMMT/C15A/C30B gradients, and treated/untreated pairs must be expanded into individual sample objects whenever the formulation rule is provided.
9. If a table reports zero loading for a component in a sample, omit that component as an entity unless it is useful as a matrix/control definition. Do not create additive entities with zero loading.
10. If a formulation table has clear samples but no performance in this paper, still extract the formulations. Performance coverage is irrelevant to this formulation task.

[SOURCE SCOPE RULES - STRICT]
1. Prefer original experimental samples prepared or tested in this paper.
2. For review papers, introduction/background comparisons, literature-summary tables, or examples cited from other papers, extract a sample only when the source gives a clear sample-level formulation for an EVA/PE/POE-family material. Otherwise do not create formulation samples from narrative literature discussion.
3. Do NOT treat generic material examples, mechanism illustrations, cited prior-work formulations, or non-tested schematic labels as formulation samples.
4. If the paper studies several polymer families, keep matrix_system specific to each sample and use "other" for non-EVA/PE/POE-family samples rather than forcing them into EVA/PE/POE.

[CONTROLLED VOCABULARIES]
matrix_system: EVA, PE, EVA/PE, EVA/POE, EVA/PE/POE, other
formulation_basis: wt_total, wt_polymer_blend, phr_resin, parts_by_weight, unknown
entity_role:
  matrix_polymer
  main_flame_retardant
  synergistic_flame_retardant
  functional_filler
  compatibilizer
  crosslinking_agent
  co_crosslinking_agent
  processing_aid
  other_additive
entity_type:
  pristine
  surface_treated
  grafted_additive
  core_shell
  microencapsulated
  coated
  intercalated
  doped
  in_situ_grown
  hybrid_complex
  matrix_grafted
  blend_package
  unknown
entity_material_type: 1 for polymer-based entity, 0 for non-polymer entity, # if unclear
part_role:
  base
  core
  shell
  coating_layer
  grafted_moiety
  grafted_backbone
  dopant
  intercalated_guest
  surface_ligand
  coupling_agent
  complexing_agent
  synergistic_part
  modifier
  unknown
modification_method:
  none
  surface_treatment
  grafting
  coating
  encapsulation
  microencapsulation
  intercalation
  doping
  in_situ_growth
  coordination_complex
  physical_adsorption
  unknown
part_relation_to_entity: part_of, core_of, shell_of, grafted_to, grafted_from, coated_on, encapsulates, doped_into, intercalated_into, complexed_with, treated_surface_of, unknown
PE_type: LDPE, LLDPE, HDPE, mPE, UHMWPE, PE, #
POE_type: ethylene-octene, ethylene-butene, POE, #
mixing_method: 0 simple blending; 1 encapsulation/surface modification; 2 reactive grafting/crosslinking; 3 masterbatch/compatibilized blending; # unknown

[MATRIX SYSTEM RULES - STRICT]
1. matrix_system describes ONLY the polymer matrix of the specific sample, not the paper title, additive package, flame retardant, compatibilizer, or other tested samples.
2. Use EVA only when EVA is the only EVA/PE/POE-family matrix polymer with nonzero matrix fraction.
3. Use PE only for pure PE/LDPE/LLDPE/HDPE/mPE/UHMWPE matrix samples with no EVA and no POE.
4. Use EVA/PE only when EVA and PE/LDPE/LLDPE/HDPE/mPE/UHMWPE are both present in the same sample matrix and POE is absent.
5. Use EVA/POE only when EVA and POE are both present in the same sample matrix and PE is absent.
6. Use EVA/PE/POE only when EVA, PE, and POE are ALL present in the same sample matrix with nonzero fractions. Do NOT use EVA/PE/POE for LLDPE/EVA, LDPE/EVA, pure LLDPE, or PS/SMA.
7. Do not infer POE from the generic word "polyolefin". Use POE only when POE, polyolefin elastomer, ethylene-octene, ethylene-butene, Engage, Exact, or similar POE resin is explicitly part of the matrix.
8. For PS, ABS, PC/ABS, PP, EPDM, NBR, PS/SMA, or other non-EVA/PE/POE matrices, use other unless EVA, PE, or POE are explicitly matrix polymers in that exact sample.
9. EVA_polymer_fraction, PE_polymer_fraction, and POE_polymer_fraction must be consistent with matrix_system. If a polymer is absent, use 0 or #. Never output EVA/PE/POE when POE_polymer_fraction is 0 or #.
10. Decide matrix_system in this order for each sample: identify the sample's matrix polymers from the formula table/text, fill EVA/PE/POE fractions, then choose the controlled label from those nonzero fractions. Do not choose a broader label just because the paper discusses several polymer families.
11. Examples: LLDPE or PE -> PE; LLDPE/EVA or PE/EVA -> EVA/PE; EVA alone -> EVA; EVA/POE -> EVA/POE; PS/SMA, PP, ABS, PC/ABS, EPDM, PA-6 without EVA/PE/POE matrix polymer -> other.

[ENTITY ROLE RULES - STRICT]
1. matrix_polymer:
   EVA, PE, LDPE, LLDPE, HDPE, mPE, POE, EPDM, PP, and other polymers forming the continuous matrix.
   EVA/PE/POE blend polymers are NOT flame retardants even if they influence combustion.
2. main_flame_retardant:
   The primary flame-retardant ENTITY actually added to the sample.
   Examples: MDH, MH, ATH, APP, MCA, MPP, red phosphorus, EG, IFR, DOPO-g-APP, KH550-MDH, APP@SiO2, microencapsulated APP, MH@Fe-GA.
3. synergistic_flame_retardant:
   An independently added ENTITY that improves flame retardancy, char formation, smoke suppression, or anti-dripping together with the main flame retardant.
   Examples: ZB, LDH, clay/MMT, silicone, POSS, CNT/GO/MXene, metal oxides, charring agents, smoke suppressants.
4. functional_filler:
   A filler primarily used for reinforcement, conductivity, thermal conductivity, or morphology control. Use synergistic_flame_retardant instead if the paper explicitly treats it as an FR synergist.
5. compatibilizer:
   PE-g-MA, EVA-g-MA, POE-g-MA, maleic-anhydride-grafted polyolefin, or similar interface agents.
6. crosslinking_agent and co_crosslinking_agent:
   DCP, BIPB, peroxide, silane crosslinker, TAIC, triallyl cyanurate, radiation crosslinking helpers.
7. processing_aid and other_additive:
    Extract minor additives when the formulation gives them, even if they are not flame retardants.
    Processing aids include paraffin wax, lubricant, calcium/zinc stearate, processing stabilizer, dispersant, plasticizer, and similar compounding aids.
    Other additives include antioxidants and stabilizers such as Irganox 1010, Irgafos 168, Irganox MD1024, HALS, UV stabilizers, pigments, and anti-aging agents.
    Do NOT hide Irganox 1010, Irgafos 168, paraffin wax, DDA, DCP, TAIC, or similar minor additives only in formulation_notes when their sample-specific loading is reported. They must be separate entities with their own loading.
8. EVA/PE cable-compound interface agents:
   PE-g-MA, PEgMA, LDPE-g-MAH, ULDPE-g-MAH, EVA-g-MA, EVAgMA, mEVA, Fusabond, Lotader, maleated EVA/PE/POE, and anhydride-modified ethylene copolymers are compatibilizers/coupling agents unless the paper explicitly uses them as the continuous matrix.
   If they are separately dosed at 1-10 wt%, 1-10 phr, or similar small amounts, record them as separate compatibilizer entities with their own loading, not as part of the base EVA fraction.
9. Crosslinking and curing:
   DCP, BIPB, peroxide, TMPTA, TAIC, TAC, triallyl cyanurate, vinyl silane, radiation crosslinking helpers, and electron-beam/radiation conditions must be captured explicitly when sample-specific.
   DCP/peroxide is crosslinking_agent; TAIC/TMPTA/TAC is co_crosslinking_agent unless the paper defines otherwise.

[CHEMICAL AND PHYSICAL DISAMBIGUATION RULES - STRICT]
1. Do NOT confuse the polymer matrix with a flame-retardant entity.
   EVA, PE, POE, LDPE, LLDPE, HDPE, PC, ABS, PP, EPDM, PA, and similar matrix polymers remain matrix_polymer unless the paper explicitly uses a separately added polymeric additive as a flame-retardant entity.
2. Do NOT classify a compatibilizer, coupling agent, surface treatment reagent, initiator, catalyst, solvent, curing aid, processing aid, lubricant, antioxidant, or pigment as retardant1/retardant2 unless it is independently added and the paper explicitly attributes flame-retardant/synergistic function to it.
3. Trade names are not chemistry. If the paper gives both a trade name and a chemical identity, put the chemical identity in entity_name and the trade name in entity_trade_name. If only the trade name is known, keep it but note the uncertainty.
4. A synthesized or treated material is a formulation entity only if it is the actual final additive introduced into the polymer. Synthesis precursors, monomers, solvents, catalysts, washing agents, acids/bases, and unreacted reagents are NOT separate formulation entities.
5. Grafting usually creates a new chemical entity. If DOPO is grafted onto APP, MDH, PE, EVA, POE, or another backbone before compounding, the final grafted product is the entity; DOPO is a part/grafted_moiety unless separately added as a free flame retardant.
6. Surface treatment and coating usually create a modified particle, not two independent additives. KH550-treated MDH, stearic-acid-treated MDH, APP@SiO2, microencapsulated APP, and metal-polyphenol-coated MH should be one entity with parts, unless the base and modifier are separately added into the polymer blend.
7. Core-shell, coated, encapsulated, intercalated, doped, and in-situ-grown materials must be decomposed into parts, but the downstream retardant slot should still receive the final added entity name. Preserve core/shell/coating/graft ratios whenever reported.
8. If a material can be either a functional filler or a synergistic flame retardant, decide from the paper's role:
   - Use synergistic_flame_retardant when the text/table says it improves LOI, UL-94, char, smoke suppression, anti-dripping, pHRR, THR, or fire residue.
   - Use functional_filler when it is only reinforcement, conductivity, thermal conductivity, rheology control, coloration, or morphology control.
9. Carbon nanotubes, graphene/GO, clay/MMT, LDH, talc, zinc borate, metal oxides, silicone, POSS, MXene, and similar additives can be synergistic flame retardants only when the paper frames them as fire-retardant/smoke/char/anti-dripping components. Otherwise use functional_filler.
10. total_fr_loading is not total filler loading if some fillers are non-FR functional fillers. It must include only main_flame_retardant and synergistic_flame_retardant entities unless the paper explicitly treats the whole filler package as flame-retardant.
11. Do not infer chemical structure, element profile, morphology, or modification method from a common abbreviation if the paper uses that abbreviation differently. Prefer the paper's own definition and mention ambiguity in notes.

[SAMPLE NAME AND TABLE CONTEXT RULES - STRICT]
1. Sample names must remain unique within one article when the paper table uses repeated short labels under different parent groups.
2. If a table has parent headers such as "FR 10 phr" and "FR 15 phr", and child rows repeat as "+MBS (1 phr)", output names that include the parent condition, e.g. "FR (10 phr) +MBS (1 phr)" and "FR (15 phr) +MBS (1 phr)".
3. If a formulation row is defined by a table caption, parent column, footnote, or surrounding paragraph, include that context in sample_name or formulation_notes so the downstream performance task can align the sample unambiguously.
4. Do NOT merge repeated child labels from different formulation blocks into one sample. They are distinct samples if the parent loading, matrix, flame retardant loading, compatibilizer loading, or treatment condition differs.
5. When sample names encode quantities, decode them only if the paper defines the code. Examples: "40IFR" means 40 wt% IFR only if the paper/table defines formulations by mass fraction; "1BAC" means 1 wt% BAC only if the table/header confirms it; "A5/A6" treatment and loading must follow the A-series rule in the paper.
6. If OCR confuses zero and letter O in sample labels, preserve the paper label as visible but use formulation_notes to mention the likely table context. Do not merge A0B and AOB unless the table context proves they are the same.

[MODIFIED / GRAFTED / CORE-SHELL ENTITY RULES]
1. If a material is synthesized, treated, coated, grafted, encapsulated, intercalated, doped, coordinated, or grown first and then added to EVA/PE/POE, output the final material as ONE entity.
2. Decompose that entity into parts whenever the paper discloses its constituents.
3. For surface treatment:
   - Entity: "KH550-treated MDH", "m-MDH", "SA-MDH", etc.
   - Parts: MDH as base; KH550/stearic acid/etc. as coupling_agent, surface_ligand, or modifier.
   - Method: surface_treatment.
4. For grafting:
   - Entity: final grafted product such as DOPO-g-APP, DOPO-g-MDH, PE-g-MA, EVA-g-MA.
   - Parts: backbone/base as grafted_backbone or base; grafted molecule/monomer as grafted_moiety.
   - Method: grafting.
   - Grafting usually creates a new chemical entity. Do NOT split the grafted molecule into an independent flame retardant unless it is separately added to the formulation.
5. For core-shell, coating, and encapsulation:
   - Entity: final product such as APP@SiO2, MBPAPP, microencapsulated APP.
   - Parts: core/base and shell/coating/polymer shell.
   - Preserve explicit shell/core ratio, monomer ratio, or coating amount in part_ratio.
6. For coordination or hybrid complexes:
   - Entity: final hybrid such as MH@Fe-GA or metal-polyphenol-coated MH.
   - Parts: base/core plus ligand, metal ion, or complexing component.
   - Method: coordination_complex or surface_treatment as supported by the paper.
7. For matrix-grafted systems:
   - If the grafting is on EVA/PE/POE itself and no discrete FR additive entity is formed, classify it as matrix_polymer or compatibilizer with entity_type = matrix_grafted.
   - Do NOT consume the flame-retardant slots unless the paper treats the matrix-grafted product itself as the flame-retardant entity.
8. For physical blends:
   - APP + ZB means two independent entities: APP and ZB.
   - KH550-treated MDH + ZB means two independent flame-retardant entities: KH550-treated MDH and ZB. KH550 is only a part of the first entity.

[QUANTITY AND BASIS RULES]
1. entity_loading is the ORIGINAL numeric loading as reported by the paper. Do not silently convert it.
2. entity_loading_text must preserve the exact original expression, including unit and denominator when visible, such as "60 phr", "25 wt% of total composite", "20 wt% based on EVA", or "EVA/APP/ZB = 100/30/5".
3. Put units only in entity_loading_unit or part_loading_unit:
   - wt%: mass percent
   - phr: parts per hundred resin/polymer
   - parts: formulation parts by weight
   - #: missing or unclear
4. Put denominator/basis only in entity_loading_basis:
   - wt_total: mass percent of the whole final formulation/composite; denominator includes polymer matrix plus all additives.
   - phr_resin: parts per 100 parts of EVA/PE/POE/polymer resin or polymer blend.
   - wt_polymer_blend: wt% based on polymer blend/resin only; denominator excludes flame retardants and other additives. This is NOT total-composite wt%.
   - parts_by_weight: formulation table gives relative mass parts such as EVA/APP/ZB = 100/30/5.
   - unknown: denominator cannot be determined.
5. entity_loading_wt_total is the normalized loading in wt% of the whole final formulation/composite. Fill it only when the paper explicitly reports it or when the complete formula makes conversion unambiguous. Otherwise use "#".
6. entity_loading_phr_resin is the normalized loading in phr, i.e. parts per 100 parts polymer resin/blend. Fill it only when explicitly reported or unambiguously computable. Otherwise use "#".
7. entity_loading_conversion_note must briefly state the denominator and conversion evidence, e.g. "reported as 60 phr; total additive phr known" or "reported as total-composite wt%". Use "#" if no conversion was made.
8. Do not convert phr to wt_total, or wt_total to phr, unless the complete denominator is known. The Python post-processor may perform conservative conversions later.
9. EVA_polymer_fraction, PE_polymer_fraction, and POE_polymer_fraction are fractions within the polymer blend only; they should sum to 100 when all are known.
10. EVA_VA_content is the vinyl acetate content inside EVA, not the EVA loading in the formulation.
11. total_fr_loading excludes EVA/PE/POE matrix polymers and should include flame-retardant package entities only. Its basis must follow formulation_basis.
12. part_loading is the normalized wt% of this part/modifier in the whole final composite/formulation. When part_loading is numeric, part_loading_unit must be wt%. Do NOT put phr, shell/core ratio, modifier/base ratio, grafting ratio, or synthesis feed ratio directly into part_loading.
13. For modified, coated, grafted, in-situ-grown, doped, or core-shell flame-retardant entities, convert modifier/part amount to final-composite wt% only when the denominator is explicit or unambiguously computable from the complete formulation. If the paper reports a part amount directly in the final formulation, use it directly, e.g. "49.5 wt% LDH + 0.5 wt% SiO2" means SiO2 part_loading = 0.5 and part_loading_unit = wt%.
14. If the paper reports only an internal modifier/base ratio, shell/core ratio, grafting ratio, monomer ratio, or synthesis feed ratio and it cannot be safely converted to final-composite wt%, set part_loading = "#", part_loading_unit = "#", and preserve the original expression in part_ratio and part_notes.

[IMAGE AND BBOX RULES]
Due to extraction, one figure may be split into multiple image files. Do NOT assume a 1:1 mapping between image IDs and captions.
For entity_fig_locator, use the best visible chemical structure, reaction scheme, synthesis scheme, or labeled particle/surface-treatment schematic for the FINAL entity actually added to the formulation.
For grafted or newly synthesized flame retardants, entity_fig_locator must point to the final grafted/synthesized product or its repeating unit when visible. Do NOT point only to isolated precursors if the final product is not shown.
For surface-treated, coated, encapsulated, core-shell, intercalated, doped, or in-situ-grown particles, entity_fig_locator may point to the schematic/morphology/synthesis evidence of the modified particle, even though the downstream visual script may correctly decide that no single molecular SMILES exists.
For part_fig_locator, use the visible structure, reaction scheme region, formula, or labeled subregion for that part if it can be isolated. This can be a shell material, ligand, coupling agent, grafted monomer/moiety, core component, or modifier.
Do NOT use part_fig_locator to imply that the final grafted product can be reconstructed from separate monomer images. The part locator is evidence for the part only; the final entity needs its own locator if the final structure is visible.
Required locator format: Fig. 2a | ID: xxx.png | short caption | target description
Use [xmin, ymin, xmax, ymax] with values between 0.000 and 1.000. Return "#" if no reliable visible region exists.

[FIGURE LOCATOR NEGATIVE RULES - STRICT]
1. The image ID must correspond to the actual panel/image that contains the target. Do NOT use a nearby image ID only because the surrounding paragraph discusses the target.
2. Do NOT label SEM, TEM, FE-SEM, AFM, optical micrographs, EDS/EDX maps, particle-size plots, XRD, FTIR, Raman, XPS, NMR, DSC, TGA/DTG curves, cone calorimeter curves, smoke-density curves, LOI/UL-94 plots, tensile curves, residue photographs, or char photographs as chemical structure or synthesis images.
3. For newly synthesized or grafted flame retardants, if the paper only shows spectra, thermal curves, morphology, performance curves, or residue images, set entity_fig_locator to "#". Do not pretend those images are final structures.
4. If only precursor molecules are shown and the final grafted/synthesized product is not visible, entity_fig_locator must be "#"; the precursor may be recorded only in part_fig_locator.
5. For surface-treated, coated, encapsulated, core-shell, intercalated, doped, or in-situ-grown particles, a morphology locator is allowed only as morphology evidence. The target description must explicitly say "morphology evidence", "particle schematic", or "surface-treatment evidence"; never call it a molecular structure or SMILES-ready structure.
6. If a caption says Fig. 1a is a synthesis route and Fig. 1b is TEM/TGA/curve, choose the image ID for Fig. 1a only. If the correct panel ID cannot be identified, output "#".
7. The target description must state what the image really is: "chemical structure", "reaction scheme", "synthesis route", "particle schematic", "morphology evidence", or "#". Do not infer a structure label from the text if the actual image is a curve or micrograph.

[ANTI-HALLUCINATION]
Use "#" rather than inferring hidden formulation details.
If the role is unclear, choose the most conservative role and explain briefly in formulation_notes or part_notes.
If exact loading is missing, output "#"; do not infer from naming such as MDH60 unless the paper defines it.

JSON schema:
{
  "article_title": "paper title",
  "DOI": "paper DOI",
  "innovation": "paper innovation, concise",
  "ai_summary": "short paper summary. Must explicitly include the journal name; if the journal cannot be identified from the paper, write 'Journal: #'.",
  "samples": [
    {
      "sample_name": "Sample identifier/name exactly as written. Preserve hyphens, slashes, ratios, suffixes, and parent table context. If repeated short labels occur under different parent groups, include the parent condition so names remain unique.",
      "matrix": "Matrix material only, using standard abbreviations such as EVA, LDPE/EVA, EVA/POE, EVM/EPDM. Do NOT include flame retardants, fillers, compatibilizers, antioxidants, crosslinkers, or coating/shell materials here.",
      "matrix_system": "Controlled matrix family for this exact sample: EVA, PE, EVA/PE, EVA/POE, EVA/PE/POE, or other. Must be based on the sample formulation, not the paper title or other samples.",
      "formulation_basis": "Denominator of the formulation data: wt_total, wt_polymer_blend, phr_resin, parts_by_weight, or unknown. This controls whether loading values are total-composite wt%, polymer-basis wt%, phr, or relative parts.",
      "EVA_polymer_fraction": "Numeric %, EVA fraction inside the polymer blend only. This is NOT EVA wt% in the whole composite. Use # if the polymer-blend composition is not specified.",
      "PE_polymer_fraction": "Numeric %, PE/LDPE/LLDPE/HDPE/mPE fraction inside the polymer blend only. Use 0 or # when absent; do not infer PE from generic polyolefin.",
      "POE_polymer_fraction": "Numeric %, POE fraction inside the polymer blend only. Use POE only when polyolefin elastomer/ethylene-octene/ethylene-butene/Engage/Exact or equivalent is explicitly a matrix polymer.",
      "EVA_VA_content": "Vinyl acetate content inside EVA/EVM, numeric wt% only. Do NOT put EVA loading, additive loading, oxygen index, or acetate-related reagent content here.",
      "PE_type": "LDPE, LLDPE, HDPE, mPE, UHMWPE, PE, or #. Extract resin density/type only when stated.",
      "POE_type": "ethylene-octene, ethylene-butene, POE, or #. Extract comonomer/type only when stated.",
      "key_content": "Matrix-specific data not covered by structured fields: EVA VA content/trade grade, PE density/type, POE comonomer, rubber/elastomer type, crosslinking/vulcanization/radiation/gel-content/crosslink-density information, cable grade, foam/expanded state, crystallinity, or other matrix features. Do NOT use this field for flame-retardant identities that should be entities.",
      "melt_index": "Melt flow index/rate of the polymer matrix or blend, numeric only when reported in g/10min or clearly equivalent. Do NOT put viscosity, torque, molecular weight, or processing temperature here.",
      "matrix_trname": "Matrix trade name/grade/manufacturer summary, e.g. Elvax 460, Levapren 450, Engage. Use # if not provided. Do NOT put flame-retardant trade names here.",
      "total_fr_loading": "Total flame-retardant package loading, numeric only. Include main_flame_retardant and synergistic_flame_retardant entities only. Exclude EVA/PE/POE matrix, compatibilizers, antioxidants, crosslinkers, catalysts, solvents, and non-FR fillers unless the paper explicitly treats the whole filler package as FR.",
      "mixing_method": "Controlled code: 0 simple melt/solution blending; 1 pre-encapsulation/surface modification/coating before blending; 2 reactive grafting/crosslinking/vulcanization/radiation curing; 3 masterbatch/compatibilized blending; # unknown. Choose the highest specific method supported by the paper.",
      "formulation_notes": "Concise notes for ambiguous basis, parent table context, minor additives, conversions, crosslinking/radiation conditions, or conservative decisions. Do not store primary structured values only in notes when a field exists.",
      "entities": [
        {
          "entity_id": "Short stable ID such as E1 within this sample. Use the same ID when the same final entity is referenced in parts.",
          "entity_name": "Final material actually added into the formulation. For modified/grafted/coated/core-shell materials, use the final product name such as DOPO-g-APP, KH550-treated MDH, APP@SiO2, or microencapsulated APP. Do NOT list synthesis precursors, solvents, catalysts, or washing agents as entities unless independently added to the polymer.",
          "entity_role": "Controlled role. matrix_polymer is the polymer matrix; main_flame_retardant is the primary FR; synergistic_flame_retardant is an independently added FR/smoke/char/anti-dripping synergist; functional_filler is reinforcement/conductivity/morphology filler; compatibilizer/crosslinking_agent/co_crosslinking_agent/processing_aid/other_additive are non-FR roles unless the paper explicitly says otherwise.",
          "entity_type": "Controlled structural state: pristine, surface_treated, grafted_additive, core_shell, microencapsulated, coated, intercalated, doped, in_situ_grown, hybrid_complex, matrix_grafted, blend_package, or unknown. This describes the final added entity, not individual precursors.",
          "entity_loading": "Original numeric loading as reported. Do not silently convert. Use # if no sample-specific loading is reported.",
          "entity_loading_text": "Exact original loading expression including unit and denominator when visible, e.g. 60 phr, 25 wt% of total composite, 20 wt% based on EVA, EVA/APP/ZB = 100/30/5. Use # if missing.",
          "entity_loading_unit": "Unit of entity_loading only: wt%, phr, parts, or #. Do not mix denominator text into this field.",
          "entity_loading_basis": "Denominator/basis only: wt_total, wt_polymer_blend, phr_resin, parts_by_weight, or unknown. wt_total includes all matrix and additives; phr_resin is per 100 parts polymer resin/blend.",
          "entity_loading_wt_total": "Normalized wt% of the whole final composite, numeric only. Fill only when explicitly reported or unambiguously computable from a complete formula. Otherwise #.",
          "entity_loading_phr_resin": "Normalized phr relative to 100 parts polymer resin/blend, numeric only. Fill only when explicitly reported or unambiguously computable. Otherwise #.",
          "entity_loading_conversion_note": "Brief evidence for denominator and conversion, e.g. reported as 60 phr relative to 100 parts EVA, or converted from EVA/APP/ZB = 100/30/5. Use # if no conversion.",
          "entity_material_type": "1 for polymer-based entity such as APP, IFR polymer, polymer shell, rubber, or polymeric charring agent; 0 for inorganic/small-molecule/particle entities; # if unclear.",
          "entity_trade_name": "Trade name or commercial grade of this entity only. Use # if not provided. Do NOT put matrix trade names here.",
          "entity_profile": "CRITICAL for NLP embedding. Use exactly: Elements: [...]. Groups: [...]. Morphology: [...]. Modification: [...]. Elements are core elements such as P/N/Si/Mg/Al/B/Zn/Br/Cl/C; Groups are key chemical structures such as DOPO, triazine, phosphate, phosphinate, hydroxide, borate, siloxane, aromatic, none; Morphology is physical form such as powder, platelet, nanotube, nanosheet, fiber, liquid, core-shell particle; Modification is surface_treated/grafted/coated/intercalated/none. Never describe synthesis steps here.",
          "entity_fig_locator": "If a real structure/synthesis/particle evidence image is shown, return: Fig./Scheme panel | ID: exact image filename | short caption | target type. Target type must be chemical structure, reaction scheme, synthesis route, particle schematic, morphology evidence, or surface-treatment evidence. Use # if no reliable target image exists.",
          "entity_bbox": "Relative bounding box [xmin, ymin, xmax, ymax] for the target visual region, values 0.000-1.000. Identify one complete visual chunk; do not include labels, arrows, unrelated molecules, spectra, curves, or morphology panels unless target type is morphology evidence. Use # if unable to isolate reliably.",
          "parts": [
            {
              "part_name": "Chemically meaningful constituent inside the final modified entity, such as core/base, shell/coating, grafted moiety, grafted backbone, ligand, dopant, intercalated guest, coupling agent, or modifier.",
              "part_role": "Controlled part role. Use base/core/shell/coating_layer/grafted_moiety/grafted_backbone/dopant/intercalated_guest/surface_ligand/coupling_agent/complexing_agent/synergistic_part/modifier/unknown based on the paper's chemistry.",
              "relation_to_entity": "Controlled relation to the final entity: part_of, core_of, shell_of, grafted_to, grafted_from, coated_on, encapsulates, doped_into, intercalated_into, complexed_with, treated_surface_of, unknown.",
              "modification_method": "Method connecting this part to the entity: none, surface_treatment, grafting, coating, encapsulation, microencapsulation, intercalation, doping, in_situ_growth, coordination_complex, physical_adsorption, or unknown.",
              "part_ratio": "Explicit internal ratio, shell/core ratio, grafting ratio, monomer ratio, coating amount, feed ratio, or composition exactly as reported. This may be relative to the modified entity and is NOT necessarily final-composite wt%. Use # when not reported.",
              "part_loading": "Numeric wt% of this part/modifier in the whole final composite/formulation only. Convert to final-composite wt% only when explicit or unambiguously computable. Do NOT put phr, modifier/base ratio, shell/core ratio, grafting ratio, or feed ratio here. Use # when final-composite wt% is unclear.",
              "part_loading_unit": "Unit of part_loading. Must be wt% when part_loading is numeric; otherwise #.",
              "part_fig_locator": "Image locator for this part only if its structure/reaction region/formula/subregion is visible and identifiable. Do not imply the final grafted product can be reconstructed from separate precursor images.",
              "part_bbox": "Relative bbox for the part visual region only. Use # if no reliable isolated region exists.",
              "part_smiles": "Explicit SMILES only if the paper text provides SMILES. Do not invent SMILES from a name or figure; downstream vision/dictionary tools handle that.",
              "part_smiles_source": "paper text if an explicit SMILES string is printed; otherwise #.",
              "part_notes": "Short note for ambiguous part role, missing ratio, surface treatment evidence, or why no SMILES/locator is reliable."
            }
          ]
        }
      ]
    }
  ]
}

Paper Content:
"""


PERFORMANCE_PROMPT = """
You are a materials researcher building a standardized EVA/PE/POE flame-retardant performance database.

If the same formulation sample is tested under multiple conditions, emit one separate samples-array record per condition. Repeat the exact sample_name, keep every condition-specific record, and state the distinguishing condition in performance_notes. Never average, overwrite, or deduplicate records merely because sample_name is the same.

Extract ALL CORE sample-level flame-retardant, thermal-degradation, cone-calorimetry, and tensile performance data from the paper, including text, tables, captions, and charts.
This task is PERFORMANCE ONLY. Do NOT extract formulation roles, SMILES, BBox, chemical structure locators, or mechanism descriptions here.
Return valid JSON only. Do not output Markdown, explanations, or extra text.
Use "#" for missing values. Numeric fields must contain numeric values only.

[CRITICAL SAMPLE COVERAGE RULES]
1. Extract every tested sample that appears in performance tables, plots, or captions.
2. Preserve sample names exactly as written when no formulation roster is provided. If a formulation roster is provided, follow the formulation-guided alignment rules below.
3. If a sample is absent from a performance table, do NOT copy values from adjacent samples.
4. If exact values are only visible in a chart and can be read with reasonable confidence, extract the approximate numeric value and mention chart extraction in performance_notes.
5. If exact value is not readable, output "#".

[SOURCE SCOPE RULES - STRICT]
1. Prefer original experimental samples prepared or tested in this paper.
2. For review papers, literature-summary tables, introduction/background comparisons, or cited prior-work examples, extract performance only when the source clearly reports sample-level values for an EVA/PE/POE-family material and the sample identity can be aligned. Otherwise output "#" for unavailable metrics rather than creating uncertain samples.
3. Do NOT treat method conditions, cited examples, mechanism labels, generic material names, or figure-legend labels as tested samples unless a sample-level performance value is explicitly reported.
4. If a paper compares several polymer families, keep only the values that belong to the exact sample label being output; do not transfer values from PC, ABS, PP, EPDM, PA, or other matrices to EVA/PE/POE samples.

[FORMULATION-GUIDED ALIGNMENT RULES - STRICT]
1. A "FORMULATION SAMPLE ROSTER" may be provided before the paper content. It is retrieved from the formulation task for this exact paper.
2. Use the roster as the alignment anchor. When a performance-table sample corresponds to a roster sample, output sample_name EXACTLY as the roster sample_name.
3. Return one JSON sample object for every roster sample. Fill unavailable performance metrics with "#"; do not copy values from other samples.
4. If the paper reports performance for an extra sample not present in the roster, also output it using the exact paper label and mention "not in formulation roster" in performance_notes.
5. If a performance table uses aliases, abbreviations, or shortened labels, map them to roster names only when the evidence is clear from table order, captions, or surrounding text. Mention the source alias in performance_notes.
6. Do not rely on row order alone unless the table explicitly lists the same ordered sample series as the roster.
7. If the same shortened performance label appears under different parent groups, do NOT output the shortened label alone. Include the parent group or map to the exact roster name.
   Example: child rows "+MBS (1 phr)" under both "FR 10 phr" and "FR 15 phr" must become distinct samples such as "FR (10 phr) +MBS (1 phr)" and "FR (15 phr) +MBS (1 phr)", or exact roster names if supplied.
8. If an alias could match more than one roster sample, do NOT guess. Output the paper label with the parent context in sample_name and explain the ambiguity in performance_notes.

[OCR ERROR RECOVERY RULES - STRICT]
1. If a quantitative value appears as "17.5 0.4" in a context that clearly means "17.5 \u00b1 0.4", extract only "17.5".
2. If thermal data show corrupted Celsius units such as "450 1C", "450 0C", or "450 C", treat them as Celsius and extract "450".
3. Only apply these repairs when the table header or surrounding text clearly identifies the metric.

[DATA DISAMBIGUATION RULES - STRICT]
1. Peak vs mean:
   pHRR must be peak HRR only. Do NOT extract mean HRR.
   pSPR must be peak smoke production rate only. Do NOT extract mean SPR.
   pCOR and pCO2R must be peak CO/CO2 production rates only. Do NOT extract average/mean values.
2. Total vs rate:
   TSP means total smoke production or total smoke release.
   Do NOT confuse TSP/TSR with pSPR.
   THR means total heat release.
   Do NOT confuse total heat evolved (THE, kJ), effective heat of combustion (EHC, MJ/kg), heat release capacity (HRC), or heat release rate (HRR) with THR.
3. Time metrics:
   TTI means cone calorimeter time to ignition only.
   Do NOT confuse TTI with time to pHRR.
   Do NOT put UL-94 burning time, afterflame time, dripping time, Epiradiateur first ignition time (IT), mean ignition period (IP), oxygen-index burning time, or decomposition onset time into TTI. Mention non-cone ignition times in performance_notes only.
4. Char residue:
   TGAresidue is from TGA only.
   residue_rate is from cone calorimetry or fire residue only.
   Do NOT put combustion char yield/CY into TGAresidue. Do NOT put TGA char residue into cone/fire residue_rate.
5. Mechanical:
   tensile_strength and elongation_at_break must be tensile-test values.
   Do NOT extract flexural, compressive, impact, tear, or hardness values into tensile fields.
   In multi-row tables, align mechanical columns by their own headers and units. "stress (MPa)" or "tensile stress (MPa)" maps to tensile_strength; "elongation at break (%)" maps to elongation_at_break.
   Do NOT shift fire-test columns such as IT(s), IP(s), burning time, or ignition period into tensile_strength or elongation_at_break.
6. UL-94:
   Encode V-0 as 0, V-1 as 1, V-2 as 2, NR/HB/no rating as 3.
7. TGA:
   T5/T10/T50 are weight-loss temperatures.
   Tmax1/Tmax2 are derivative thermogravimetry peak temperatures.
   If several atmospheres are reported, prefer nitrogen unless the paper clearly focuses on air/oxygen; mention the atmosphere in performance_notes.
   TGAresidue should prefer the residue at 500-600 deg C when multiple residue temperatures are reported. If only final residue at a much higher temperature is reported, use it only when the paper defines it as the relevant TGA residue and note the temperature.
8. Cone calorimeter:
   Extract cone_heat_flux and cone_sample_thickness when explicitly provided.
   pHRR unit is kW/m2; THR unit is MJ/m2; TSP unit is m2; residue_rate unit is %.
   cone_sample_thickness means specimen thickness only. Do NOT extract thermocouple depth, char depth, degradation-front depth, or sample width/length into this field.
9. Test-condition values:
   cone_heat_flux and cone_sample_thickness are test conditions, not material performance targets. Extract them only when they are explicitly the external cone/MCC/fire-test heat flux or specimen thickness applying to that sample/table/figure.
   A number such as 35 or 50 kW/m2 is usually cone_heat_flux, never pHRR, THR, TTI, LOI, or residue.
   Do NOT copy a cone_heat_flux from a methods section onto samples that are not part of that cone/fire-test table or figure.
10. Curves and captions:
   Values in TGA/DTG/DSC/XRD/FTIR/Raman/XPS/NMR, SEM/TEM captions, smoke-density curves, residue photographs, or morphology figures are not LOI, UL-94, pHRR, THR, pSPR, TSP, pCOR, pCO2R, or cone residue unless the caption/table explicitly reports that metric.
   Smoke density, optical density, transmittance, absorbance, or SPR curve axis values must not be converted into TSP or pSPR unless the axis/header explicitly names total smoke production/release or peak smoke production rate.

[EXPERIMENT AND UNIT NORMALIZATION RULES - STRICT]
1. First verify the experiment and physical quantity. Then normalize units.
   A value may be converted only when it comes from the correct experiment/metric and the conversion is dimensionally valid from the paper's unit to the schema unit.
   If the experiment/metric is different, output "#" even if the unit looks convertible.
2. Always put the normalized numeric value in the JSON field. Mention the original unit/value and conversion in performance_notes when conversion was made.
3. pHRR: use peak heat-release rate from cone calorimeter or MCC only when clearly reported as a peak HRR/PHRR/pkRHR value.
   Normalize W/m2, kW/m2, MW/m2, W/cm2, or kW/m^2 to kW/m2.
   Do NOT use HRR plateaus, average HRR, total heat, EHC, or HRC as pHRR.
4. THR: use total heat release from the corresponding fire test only, normalized to MJ/m2.
   Convert J/m2, kJ/m2, MJ/m2, J/cm2, or kJ/cm2 to MJ/m2.
   If the paper reports total heat evolved/total heat as J or kJ without area normalization, convert to MJ/m2 only when exposed specimen area is explicitly known; otherwise output "#".
   Do NOT use EHC (MJ/kg), HRC, or HRR as THR.
5. pSPR: use peak smoke production rate only, normalized to m2/s. Convert cm2/s to m2/s if needed.
   Do NOT put VSP in m3/s, smoke density, light transmission, or total smoke into pSPR.
6. TSP: use total smoke production/release only, normalized to m2. Convert cm2 to m2 if needed.
   Do NOT put smoke production rate, VSP in m3/s, smoke density, or light-transmission values into TSP.
7. pCOR and pCO2R: use peak CO/CO2 mass production rate only, normalized to mg/s.
   Convert g/s, kg/s, ug/s, mg/min, or g/min to mg/s.
   Do NOT convert gas volume flow (m3/s, L/s), concentration (ppm), yield (kg/kg), or average CO/CO2 into mg/s unless the paper explicitly supplies all conversion conditions and formula.
8. TTI: use time to ignition from the corresponding cone/fire test only when it is a true ignition-time metric. Normalize min or ms to seconds.
   Do NOT put UL-94 afterflame time, time to pHRR, decomposition time, or mean ignition period into TTI.
9. cone_heat_flux: normalize W/m2, kW/m2, W/cm2, or cal/cm2/s to kW/m2 when the value is the external heat flux of the cone/fire test.
10. cone_sample_thickness: normalize cm, um, inch, or mil to mm when the value is specimen thickness. Do NOT use thermocouple depth, char depth, width, or length.
11. Tensile strength: use tensile stress/strength at break or maximum tensile strength from tensile testing only. Normalize Pa, kPa, MPa, GPa, psi, or ksi to MPa.
12. Elongation at break: use tensile elongation/strain at break only. Normalize fraction to percent only when the paper clearly reports strain as a ratio; e.g. 14.06 -> 1406%.
13. TGA temperatures: use TGA/DTG temperatures only. Normalize K to deg C by subtracting 273.15; normalize deg F only if clearly Fahrenheit.
14. LOI is oxygen index percent/vol%. Normalize decimal fraction to percent only when the paper clearly writes it as a fraction of oxygen index; e.g. 0.31 -> 31. Do NOT put sample loading percent, VA content, chlorine content, oxygen concentration used for a char test, or phosphorus content into LOI.
15. UL94_rating must encode only UL-94 vertical/horizontal rating. Do NOT encode LOI, NC in unrelated tests, Epiradiateur classification, or "burning" in another test unless it is clearly the UL-94 result.

[FIELD-SPECIFIC NEGATIVE EXAMPLES - STRICT]
1. Do NOT put "35 kW/m2" or "50 kW/m2" into pHRR. They are external heat flux conditions and belong only in cone_heat_flux.
2. Do NOT put cone/MCC sample mass, sample dimensions, thickness, heating rate, gas-flow rate, irradiation level, or test temperature into LOI, TGA, pHRR, THR, smoke, CO/CO2, tensile, or residue fields.
3. Do NOT put TGA residue into residue_rate, and do NOT put cone/fire residue into TGAresidue. Keep both fields separate even if both are percentages.
4. Do NOT put time to pHRR, afterflame time, afterglow time, dripping time, IP/IT from Epiradiateur, degradation onset time, or temperature-at-peak into TTI.
5. Do NOT put MFI, melt viscosity, torque, crystallinity, hardness, impact strength, flexural strength, compressive strength, dielectric loss, or volume/surface resistance into tensile_strength or elongation_at_break.
6. If a row/column header is corrupted by OCR and the metric cannot be identified from neighboring headers and units, output "#" and mention the ambiguity in performance_notes.
7. Non-core cable, processing, electrical, aging, and oxidation-stability metrics are outside this main task:
   MFI/melt flow rate, equilibrium torque, hot-set, gel content, crosslink density, hardness, smoke density, acid gas release, OIT, OITP, volume resistivity, dielectric strength, aged tensile strength, and aged elongation.
   Do NOT extract them into structured fields in this task. Do NOT put their values in performance_notes. Ignore them.

[ANTI-HALLUCINATION]
Never synthesize missing metrics.
Never assume all samples were tested for all metrics.
Never use values from neat matrix or control samples for modified samples unless the sample names match exactly.
Never repair incompatible experiments or incompatible dimensions by guesswork.
Use "#" when uncertain.

JSON schema:
{
  "article_title": "paper title",
  "DOI": "paper DOI",
  "innovation": "paper innovation, concise",
  "ai_summary": "short paper summary. Must explicitly include the journal name; if the journal cannot be identified from the paper, write 'Journal: #'.",
  "samples": [
    {
      "sample_name": "Sample identifier/name. If a formulation roster is provided and alignment is clear, output the exact roster sample_name. Otherwise preserve the paper label and parent table context. Do not guess aliases when ambiguous.",
      "LOI": "Limiting oxygen index, numeric percent/vol% only. Do NOT put sample loading %, VA content, oxygen concentration used for another test, phosphorus content, or char oxygen content here. Use # if not explicitly LOI/OI.",
      "UL94_rating": "UL-94 vertical/horizontal rating only: 0 for V-0, 1 for V-1, 2 for V-2, 3 for NR/HB/no rating. Do NOT encode LOI, burning time, dripping time, Epiradiateur class, or non-UL-94 terms.",
      "elongation_at_break": "Tensile elongation/strain at break, numeric % only. Must be from tensile testing. Do NOT extract flexural, compressive, impact, tear, hardness, or DMA values.",
      "tensile_strength": "Tensile stress/strength at break or maximum tensile strength, numeric MPa only. Must be tensile. Do NOT extract compressive, flexural, impact, tear, modulus, hardness, or torque values.",
      "T1": "TGA 1% weight-loss temperature, numeric deg C only. Use only TGA/thermal-degradation data.",
      "T5": "TGA 5% weight-loss temperature, numeric deg C only. Do NOT use processing temperature, melting temperature, onset from unrelated tests, or cone ignition time.",
      "T10": "TGA 10% weight-loss temperature, numeric deg C only. Use # if not reported.",
      "T50": "TGA 50% weight-loss temperature, numeric deg C only. Do NOT infer from a curve unless readable with reasonable confidence and noted.",
      "Tmax1": "First DTG/TGA maximum degradation peak temperature, numeric deg C only. Do NOT use DSC melting/crystallization peaks.",
      "Tmax2": "Second DTG/TGA maximum degradation peak temperature, numeric deg C only. Use # if absent.",
      "TGAresidue": "TGA char/residue yield, numeric % only. Prefer 500-600 deg C residue when multiple temperatures exist. Do NOT put cone residue, fire residue, mass remaining after UL-94, or char photograph estimates here.",
      "TTI": "Time to ignition from cone calorimeter/fire test only, numeric seconds. Do NOT put time to pHRR, afterflame time, afterglow time, dripping time, IP/IT from Epiradiateur, oxygen-index burning time, or TGA onset time here.",
      "cone_heat_flux": "External heat flux used in cone/MCC/fire calorimetry, numeric kW/m2, e.g. 35 or 50. This is a test condition, not material performance. Do NOT put it into pHRR/THR/TTI/LOI/residue.",
      "cone_sample_thickness": "Specimen thickness for the cone/fire test, numeric mm. Do NOT extract sample width/length, thermocouple depth, char depth, coating thickness, or particle size.",
      "pHRR": "Peak heat release rate from cone calorimeter or MCC, numeric kW/m2 only. MUST be peak value. Do NOT extract mean HRR, HRC, EHC, total heat, external heat flux, or HRR curve axis values without a peak sample value.",
      "THR": "Total heat release from the corresponding fire test, numeric MJ/m2 only when area-normalized or unambiguously convertible. Do NOT use EHC, THE in kJ without area, HRC, or HRR.",
      "pSPR": "Peak smoke production rate, numeric m2/s only. MUST be peak rate. Do NOT extract mean SPR, smoke density, optical density, light transmission, VSP, or TSP/TSR.",
      "TSP": "Total smoke production/release, numeric m2 only. Do NOT put peak smoke production rate, smoke density, optical density, transmittance, VSP, or smoke curve axis values here.",
      "pCOR": "Peak CO production mass rate, numeric mg/s only. Do NOT extract mean/average CO rate, CO yield, CO concentration ppm, or gas volume flow unless the paper explicitly gives a valid conversion.",
      "pCO2R": "Peak CO2 production mass rate, numeric mg/s only. Do NOT extract mean/average CO2 rate, CO2 yield, CO2 concentration, or gas volume flow.",
      "residue_rate": "Cone/fire-test residue, char residue, or mass residue after combustion, numeric % only. Do NOT put TGA residue here; keep TGAresidue and fire residue separate.",
      "performance_notes": "Short notes for source table/figure, sample alias, test atmosphere, test condition, unit conversion, chart-read approximation, or ambiguity. Use notes to explain why a field is # when the metric is similar but not compatible."
    }
  ]
}

Paper Content:
"""


def is_missing(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip() in {"", "#", "nan", "None", "N/A"}


def clean_value(value):
    return None if is_missing(value) else value


def normalize_exact_dedupe_value(value) -> str:
    if is_missing(value):
        return ""
    text = " ".join(str(value).strip().split())
    match = re.fullmatch(r"([wl]-)?([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", text)
    if not match:
        return text
    prefix = match.group(1) or ""
    try:
        number = float(match.group(2))
    except ValueError:
        return text
    if number.is_integer():
        return f"{prefix}{int(number)}"
    return f"{prefix}{number:.12g}"


def drop_exact_duplicate_rows(df: pd.DataFrame, columns: Optional[List[str]] = None) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    use_columns = [col for col in (columns or list(df.columns)) if col in df.columns]
    if not use_columns:
        return df, 0
    normalized = df[use_columns].apply(lambda col: col.map(normalize_exact_dedupe_value))
    keep_mask = ~normalized.duplicated(keep="first")
    removed = int((~keep_mask).sum())
    if removed == 0:
        return df, 0
    return df.loc[keep_mask].copy(), removed


def join_unique(values) -> Optional[str]:
    result = []
    seen = set()
    for value in values:
        if is_missing(value):
            continue
        text = str(value).strip()
        key = normalize_name(text) or text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return "; ".join(result) if result else None


def first_value(df: pd.DataFrame, column: str):
    if column not in df.columns:
        return None
    for value in df[column].tolist():
        if not is_missing(value):
            return value
    return None


def joined_column_values(df: pd.DataFrame, column: str) -> Optional[str]:
    if column not in df.columns:
        return None
    return join_unique(df[column].tolist())


def repair_common_utf8_mojibake(value) -> str:
    text = str(value or "")
    suspicious = any(
        character in text or any(0x80 <= ord(item) <= 0x9F for item in text)
        for character in ("\u00c3", "\u00c2", "\u00e2", "\u00f0", "\u00ef")
    )
    if not suspicious:
        return text

    def penalty(candidate: str) -> int:
        return (
            10 * candidate.count("\ufffd")
            + 4 * sum(unicodedata.category(character) == "Cc" for character in candidate)
            + 2 * sum(candidate.count(character) for character in ("\u00c3", "\u00c2", "\u00e2", "\u00f0", "\u00ef"))
        )

    best = text
    best_penalty = penalty(text)
    for _ in range(3):
        improved = False
        for encoding in ("latin-1", "cp1252"):
            try:
                candidate = best.encode(encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            candidate_penalty = penalty(candidate)
            if candidate_penalty < best_penalty:
                best = candidate
                best_penalty = candidate_penalty
                improved = True
        if not improved:
            break
    return best


def normalize_name(value) -> str:
    if is_missing(value):
        return ""
    text = unicodedata.normalize("NFKC", repair_common_utf8_mojibake(value)).casefold()
    return "".join(character for character in text if character.isalnum())


def normalize_identity_text(value) -> str:
    if is_missing(value):
        return ""
    return " ".join(
        unicodedata.normalize("NFKC", repair_common_utf8_mojibake(value)).split()
    )


def normalize_role(value) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def numeric_value(value) -> float:
    if is_missing(value):
        return -1.0
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else -1.0


def numeric_or_none(value) -> Optional[float]:
    number = numeric_value(value)
    return number if number >= 0 else None


def normalize_key_number(value) -> str:
    number = numeric_or_none(value)
    if number is None:
        return ""
    if float(number).is_integer():
        return str(int(number))
    return str(number)


def make_sample_uid(article_uid, sample_order, sample_name) -> str:
    article_key = str(article_uid or "").strip()
    order_key = normalize_key_number(sample_order) or "#"
    name_key = normalize_name(sample_name) or "#"
    return f"{article_key}::S{order_key}::{name_key}"


PERFORMANCE_CONDITION_COLUMNS = ["cone_heat_flux", "cone_specimen_thickness"]


def normalize_performance_condition_value(value) -> str:
    normalized = normalize_exact_dedupe_value(value)
    if normalized.startswith(("w-", "l-")):
        normalized = normalized[2:]
    return normalized


def build_performance_condition_signature(row: Dict) -> str:
    parts = []
    for column in PERFORMANCE_CONDITION_COLUMNS:
        value = clean_value(row.get(column))
        if value is None:
            continue
        parts.append(f"{column}={normalize_performance_condition_value(value)}")
    return "|".join(parts) if parts else "default"


def make_performance_record_uid(sample_uid, record_order, condition_signature) -> str:
    signature = str(condition_signature or "default")
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    order_key = normalize_key_number(record_order) or "1"
    return f"{sample_uid}::P{order_key}::{digest}"


def attach_stable_identity(row: Dict, file_id: str) -> None:
    article_uid = str(file_id or "").strip()
    sample_uid = make_sample_uid(article_uid, row.get("sample_index_in_article"), row.get("sample_name"))
    row["article_uid"] = article_uid
    row["sample_uid"] = sample_uid
    if "performance_record_index" in row or "performance_condition_signature" in row:
        record_order = row.get("performance_record_index") or 1
        condition_signature = row.get("performance_condition_signature") or build_performance_condition_signature(row)
        row["performance_record_index"] = record_order
        row["performance_condition_signature"] = condition_signature
        row["performance_record_uid"] = make_performance_record_uid(sample_uid, record_order, condition_signature)


def normalize_unit(value) -> str:
    text = str(value or "").strip().lower().replace(" ", "")
    text = text.replace("wt.%", "wt%").replace("weight%", "wt%").replace("mass%", "wt%")
    if text in {"wt%", "%"}:
        return "wt%"
    if text in {"phr", "php"}:
        return "phr"
    if text in {"part", "parts", "pbw"}:
        return "parts"
    return text or "#"


def normalize_basis(value) -> str:
    basis = normalize_role(value)
    aliases = {
        "total": "wt_total",
        "total_formulation": "wt_total",
        "whole_formulation": "wt_total",
        "composite": "wt_total",
        "resin": "phr_resin",
        "polymer": "phr_resin",
        "polymer_blend": "wt_polymer_blend",
        "matrix": "wt_polymer_blend",
        "parts": "parts_by_weight",
    }
    return aliases.get(basis, basis or "unknown")


def detect_matrix_components(matrix_text) -> tuple:
    text = str(matrix_text or "").lower()
    eva = bool(re.search(r"\beva\b|ethylene[- ]vinyl acetate|ethylene vinyl acetate", text))
    pe = bool(re.search(r"\b(?:pe|ldpe|lldpe|hdpe|mpe|uhmwpe)\b|polyethylene|poly\(ethylene\)", text))
    poe = bool(re.search(r"\bpoe\b|polyolefin elastomer|ethylene[- ](?:octene|butene)|engage|exact", text))
    other = bool(re.search(r"\b(?:ps|sma|abs|pc|pp|epdm|nbr|pvc|pa6|pa-6)\b|polystyrene|polypropylene", text))
    return eva, pe, poe, other


def family_matrix_system(eva: bool, pe: bool, poe: bool) -> str:
    if eva and pe and poe:
        return "EVA/PE/POE"
    if eva and pe:
        return "EVA/PE"
    if eva and poe:
        return "EVA/POE"
    if eva:
        return "EVA"
    if pe:
        return "PE"
    return "other"


def normalize_matrix_system_value(matrix_system, matrix, eva_fraction, pe_fraction, poe_fraction) -> str:
    eva_num = numeric_or_none(eva_fraction)
    pe_num = numeric_or_none(pe_fraction)
    poe_num = numeric_or_none(poe_fraction)
    has_fraction_evidence = any(value is not None for value in [eva_num, pe_num, poe_num])
    eva_text, pe_text, poe_text, other_text = detect_matrix_components(matrix)

    if has_fraction_evidence and any((value or 0) > 0 for value in [eva_num, pe_num, poe_num]):
        if other_text:
            return "other"
        return family_matrix_system(
            (eva_num or 0) > 0,
            (pe_num or 0) > 0,
            (poe_num or 0) > 0,
        )

    if eva_text or pe_text or poe_text or other_text:
        if other_text:
            return "other"
        return family_matrix_system(eva_text, pe_text, poe_text)

    value = clean_value(matrix_system)
    if value in {"EVA", "PE", "EVA/PE", "EVA/POE", "EVA/PE/POE", "other"}:
        return value
    return "other"


def entity_is_matrix(row: pd.Series) -> bool:
    return normalize_role(row.get("entity_role")) == "matrix_polymer"


def loading_value_for_sort(row: pd.Series) -> float:
    for column in ["entity_loading_wt_total_formulation", "entity_loading_phr_resin", "entity_loading"]:
        value = numeric_or_none(row.get(column))
        if value is not None:
            return value
    return -1.0


def compute_loading_conversions(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    if "entity_loading" not in group.columns:
        return group
    for column in ["entity_loading_wt_total_formulation", "entity_loading_phr_resin", "entity_loading_conversion_notes"]:
        if column not in group.columns:
            group[column] = None

    non_matrix = group[~group.apply(entity_is_matrix, axis=1)]
    loaded_non_matrix = non_matrix[non_matrix["entity_loading"].apply(lambda value: numeric_or_none(value) is not None)]
    if loaded_non_matrix.empty:
        return group

    polymer_basis_mask = loaded_non_matrix.apply(
        lambda row: normalize_basis(row.get("entity_loading_basis")) in {"phr_resin", "wt_polymer_blend"}
        and normalize_unit(row.get("entity_loading_unit")) in {"phr", "wt%"},
        axis=1,
    )
    polymer_basis_rows = loaded_non_matrix[polymer_basis_mask]
    complete_polymer_basis = len(loaded_non_matrix) > 0 and len(polymer_basis_rows) == len(loaded_non_matrix)
    polymer_basis_total = sum(numeric_or_none(v) or 0 for v in polymer_basis_rows["entity_loading"].tolist())

    wt_total_mask = loaded_non_matrix.apply(
        lambda row: normalize_basis(row.get("entity_loading_basis")) == "wt_total"
        and normalize_unit(row.get("entity_loading_unit")) == "wt%",
        axis=1,
    )
    wt_total_rows = loaded_non_matrix[wt_total_mask]
    complete_wt_total = len(loaded_non_matrix) > 0 and len(wt_total_rows) == len(loaded_non_matrix)
    wt_total_additives = sum(numeric_or_none(v) or 0 for v in wt_total_rows["entity_loading"].tolist())
    resin_wt_total = 100.0 - wt_total_additives

    parts_mask = group.apply(
        lambda row: normalize_basis(row.get("entity_loading_basis")) == "parts_by_weight"
        and normalize_unit(row.get("entity_loading_unit")) == "parts"
        and numeric_or_none(row.get("entity_loading")) is not None,
        axis=1,
    )
    parts_rows = group[parts_mask]
    complete_parts = len(parts_rows) > 0 and len(parts_rows) == len(group[group["entity_loading"].apply(lambda value: numeric_or_none(value) is not None)])
    total_parts = sum(numeric_or_none(v) or 0 for v in parts_rows["entity_loading"].tolist())
    matrix_parts = sum(
        numeric_or_none(row.get("entity_loading")) or 0
        for _, row in parts_rows.iterrows()
        if entity_is_matrix(row)
    )
    complete_parts = complete_parts and matrix_parts > 0

    for idx, row in group.iterrows():
        original = numeric_or_none(row.get("entity_loading"))
        if original is None:
            continue

        unit = normalize_unit(row.get("entity_loading_unit"))
        basis = normalize_basis(row.get("entity_loading_basis"))
        wt_total = numeric_or_none(row.get("entity_loading_wt_total_formulation"))
        phr = numeric_or_none(row.get("entity_loading_phr_resin"))
        notes = []

        if unit == "wt%" and basis == "wt_total":
            wt_total = original if wt_total is None else wt_total
            notes.append("reported as wt% of whole formulation")

        if basis in {"phr_resin", "wt_polymer_blend"} and unit in {"phr", "wt%"}:
            phr = original if phr is None else phr
            if unit == "wt%" and basis == "wt_polymer_blend":
                notes.append("reported as wt% based on polymer blend; treated as phr-equivalent")
            else:
                notes.append("reported as phr relative to 100 parts resin")
            if complete_polymer_basis and polymer_basis_total >= 0:
                wt_total = original / (100.0 + polymer_basis_total) * 100.0
                notes.append(f"converted to wt_total using total non-matrix polymer-basis loading={polymer_basis_total:g}")

        if unit == "wt%" and basis == "wt_total" and complete_wt_total and resin_wt_total > 0:
            phr = original / resin_wt_total * 100.0
            notes.append(f"converted to phr using resin fraction={resin_wt_total:g} wt%")

        if unit == "parts" and basis == "parts_by_weight":
            if complete_parts and total_parts > 0:
                wt_total = original / total_parts * 100.0
                notes.append(f"converted to wt_total using total formulation parts={total_parts:g}")
            if complete_parts and matrix_parts > 0:
                phr = original / matrix_parts * 100.0
                notes.append(f"converted to phr using matrix parts={matrix_parts:g}")

        if wt_total is not None:
            group.at[idx, "entity_loading_wt_total_formulation"] = round(wt_total, 6)
        if phr is not None:
            group.at[idx, "entity_loading_phr_resin"] = round(phr, 6)
        if notes and is_missing(row.get("entity_loading_conversion_notes")):
            group.at[idx, "entity_loading_conversion_notes"] = "; ".join(dict.fromkeys(notes))

    return group


def has_image_id(locator) -> bool:
    return "ID:" in str(locator or "")


def source_md_path(paper_dir: str) -> Optional[str]:
    target_dir = Path(paper_dir)
    auto_dir = target_dir / "auto"
    if auto_dir.exists() and auto_dir.is_dir():
        target_dir = auto_dir
    try:
        for path in target_dir.iterdir():
            if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
                return str(path)
    except Exception:
        return None
    return None


def extract_image_contexts_from_markdown(
    markdown_text: str,
    image_paths: List[str],
    before_lines: int = 10,
    after_lines: int = 10,
    max_chars: int = 2200,
) -> Dict[str, str]:
    """Return small caption/nearby-text snippets keyed by image basename."""
    if not markdown_text or not image_paths:
        return {}

    lines = markdown_text.splitlines()
    basenames = [os.path.basename(path) for path in image_paths]
    contexts: Dict[str, str] = {}

    for basename in basenames:
        hit_indices = [
            idx
            for idx, line in enumerate(lines)
            if basename in line or basename.replace(" ", "%20") in line
        ]
        if not hit_indices:
            continue

        snippets = []
        for hit_idx in hit_indices[:3]:
            start = max(0, hit_idx - before_lines)
            end = min(len(lines), hit_idx + after_lines + 1)
            window = []
            for line_no in range(start, end):
                text = lines[line_no].strip()
                if not text:
                    continue
                if len(text) > 700:
                    text = text[:700] + "..."
                window.append(f"L{line_no + 1}: {text}")
            if window:
                snippets.append("\n".join(window))

        context = "\n--- nearby context ---\n".join(snippets)
        if len(context) > max_chars:
            context = context[:max_chars] + "\n...[context truncated]"
        contexts[basename] = context

    return contexts


def append_rows_to_excel(excel_path: str, rows: List[Dict], columns: List[str], dedupe_exact: bool = False):
    if not rows:
        return
    df_to_append = pd.DataFrame(rows)
    if not os.path.exists(excel_path):
        if dedupe_exact:
            df_to_append, _ = drop_exact_duplicate_rows(df_to_append, columns)
        df_to_append.to_excel(excel_path, index=False, columns=columns)
        return
    df_existing = read_excel_canonical(excel_path)
    df_combined = pd.concat([df_existing, df_to_append], ignore_index=True)
    if dedupe_exact:
        df_combined, removed = drop_exact_duplicate_rows(df_combined, columns)
        if removed:
            print(f"[normalize] removed {removed} exact duplicate rows from {excel_path}")
    df_combined.to_excel(excel_path, index=False, columns=columns)


class SplitTaskAnalyzer(EvaPipelineBaseAnalyzer):
    task_name = "split"
    include_images = False

    def __init__(self, output_excel: str, model_name: str, input_folder: str):
        super().__init__(output_excel=output_excel, model_name=model_name, input_folder=input_folder)
        self.column_order = []

    def build_content_blocks(
        self,
        markdown_text: str,
        image_paths: List[str],
        article_idx: Optional[int] = None,
        file_id: Optional[str] = None,
    ) -> List[Dict]:
        blocks = []
        if markdown_text:
            blocks.append({"type": "text", "content": markdown_text, "page": 1})
        if self.include_images:
            for i, image_path in enumerate(image_paths):
                try:
                    with open(image_path, "rb") as f:
                        b64_image = base64.b64encode(f.read()).decode("utf-8")
                    blocks.append({
                        "type": "image",
                        "content": b64_image,
                        "page": i + 1,
                        "image_path": image_path,
                        "image_id": os.path.basename(image_path),
                        "category": "figure",
                    })
                except Exception as exc:
                    self.log(f"Failed to read image {image_path}: {exc}")
        return blocks

    def parse_task_response(self, response_text: str, article_idx: int) -> List[Dict]:
        raise NotImplementedError

    def attach_source_metadata(self, rows: List[Dict], file_id: str, paper_dir: str) -> List[Dict]:
        md_path = source_md_path(paper_dir)
        source_dir = os.path.normpath(os.path.abspath(paper_dir))
        for row in rows:
            row["source_id"] = file_id
            row["source_folder"] = source_dir
            row["source_markdown_path"] = md_path
            attach_stable_identity(row, file_id)
        return rows

    def process_paper_dir(self, paper_dir: str, article_idx: int = None):
        if not self.running:
            return

        file_id = os.path.basename(os.path.normpath(paper_dir))
        raw_path = os.path.join(self.output_dir, "api_raw", f"{self.task_name}_raw_{file_id}.json")

        if getattr(self, "force", False) and not self.dry_run:
            self._force_remove_existing_raw(raw_path, file_id)

        if os.path.exists(raw_path):
            if getattr(self, "force", False) and self.dry_run:
                self.log(f"[{self.task_name}] dry-run force requested; existing raw kept: {file_id}")
            if not self._raw_is_complete_json(raw_path):
                if self.dry_run:
                    self.run_processed += 1
                    self.run_failed += 1
                    self.log(f"[{self.task_name}] existing raw is incomplete JSON; dry-run keeps it in place: {file_id}")
                    return "failed"
                bad_path = self._move_bad_raw(raw_path, "incomplete_json")
                self.log(f"[{self.task_name}] existing raw is incomplete JSON, moved to api_raw_bad: {os.path.basename(bad_path)}")
                with self.excel_lock:
                    dropped_rows = self._drop_rows_for_file(file_id)
                if dropped_rows:
                    self.log(f"[{self.task_name}] dropped stale Excel rows before rerun: {file_id} ({dropped_rows} rows)")
            else:
                if self._raw_has_rows_in_excel(file_id):
                    self.log(f"[{self.task_name}] skipping existing raw response: {file_id}")
                    return "skipped"
                self.log(f"[{self.task_name}] existing raw has no Excel rows, trying repair: {file_id}")
                try:
                    rows = self._rows_from_existing_raw(raw_path, file_id, paper_dir, article_idx)
                    if not rows:
                        self.run_processed += 1
                        self.run_empty += 1
                        self.log(f"[{self.task_name}] existing raw is valid and empty: {file_id}")
                        return "empty"
                    with self.excel_lock:
                        self.append_to_excel(rows)
                    self.run_processed += 1
                    self.run_success += 1
                    self.log(f"[{self.task_name}] repaired from raw and saved {file_id}: {len(rows)} rows")
                    return "success"
                except Exception as exc:
                    if self.dry_run:
                        self.run_processed += 1
                        self.run_failed += 1
                        self.log(f"[{self.task_name}] raw is not recoverable; dry-run keeps it in place: {file_id}")
                        return "failed"
                    bad_path = self._move_bad_raw(raw_path, type(exc).__name__)
                    self.log(f"[{self.task_name}] raw is not recoverable, moved to api_raw_bad: {os.path.basename(bad_path)}")
        if file_id in self.processed_files:
            return "skipped"
        self.processed_files.add(file_id)

        try:
            self.log(f"[{self.task_name}] parsing paper directory: {file_id}")
            markdown_text, image_paths = self.read_mineru_output(paper_dir)
            if not markdown_text and not image_paths:
                self.run_failed += 1
                self.log(f"{file_id} contains no extractable content.")
                return "failed"

            blocks = self.build_content_blocks(markdown_text, image_paths, article_idx=article_idx, file_id=file_id)
            response_text = self.call_llm_api(blocks, file_id)
            rows = []
            if not self.dry_run and response_text:
                try:
                    rows = self.parse_task_response(response_text, article_idx)
                    rows = self.attach_source_metadata(rows, file_id, paper_dir)
                except Exception as exc:
                    bad_dir = os.path.join(self.output_dir, "api_raw_bad")
                    os.makedirs(bad_dir, exist_ok=True)
                    bad_path = os.path.join(
                        bad_dir,
                        f"{self.task_name}_raw_{file_id}.parse_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    )
                    with open(bad_path, "w", encoding="utf-8") as f:
                        f.write(response_text)
                    self.log(f"[{self.task_name}] API response parse failed; saved to api_raw_bad: {os.path.basename(bad_path)}")
                    raise exc
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(response_text)

            if rows:
                with self.excel_lock:
                    self.append_to_excel(rows)
                self.run_success += 1
                self.log(f"[{self.task_name}] saved {file_id}: {len(rows)} rows")
                return "success"
            elif not self.dry_run and response_text:
                self.run_empty += 1
                self.log(f"[{self.task_name}] {file_id} returned a valid empty result.")
                return "empty"
            else:
                self.run_failed += 1
                self.log(f"[{self.task_name}] {file_id} returned no valid data.")
                return "failed"
        except Exception as exc:
            self.run_failed += 1
            self.log(f"[{self.task_name}] fatal error: {exc}")
            traceback.print_exc()
            return "failed"
        finally:
            self.run_processed += 1

    def process_folder(self, folder_path: str):
        if not os.path.exists(folder_path):
            self.log(f"Error: folder does not exist: {folder_path}")
            return
        subdirs = sorted(d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d)))
        limit = getattr(self, "limit", 0)
        if limit and limit > 0:
            subdirs = subdirs[:limit]
        workers = max(1, int(getattr(self, "workers", 1) or 1))
        self.log(f"\n=== Starting [{self.task_name}] for {len(subdirs)} paper directories with workers={workers} ===")
        pbar = tqdm(total=len(subdirs), desc=f"{self.task_name}")
        if workers == 1:
            for idx, subdir in enumerate(subdirs, 1):
                if not self.running:
                    break
                self.process_paper_dir(os.path.join(folder_path, subdir), article_idx=idx)
                pbar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.process_paper_dir, os.path.join(folder_path, subdir), idx): subdir
                    for idx, subdir in enumerate(subdirs, 1)
                }
                for future in as_completed(futures):
                    if not self.running:
                        break
                    try:
                        future.result()
                    except Exception as exc:
                        self.run_failed += 1
                        self.log(f"[{self.task_name}] concurrent task failed for {futures[future]}: {exc}")
                    pbar.update(1)
        pbar.close()


class EntityFormulationAnalyzer(SplitTaskAnalyzer):
    task_name = "formulation_entity"
    include_images = True

    def __init__(self, output_excel: str, parts_excel: str, model_name: str, input_folder: str):
        super().__init__(output_excel=output_excel, model_name=model_name, input_folder=input_folder)
        self.parts_excel = parts_excel
        self.prompt_template = ENTITY_FORMULATION_PROMPT
        self.column_order = ENTITY_COLUMNS
        self._part_context = threading.local()

    def _get_pending_part_rows(self) -> List[Dict]:
        return getattr(self._part_context, "rows", [])

    def _set_pending_part_rows(self, rows: List[Dict]) -> None:
        self._part_context.rows = rows

    def parse_task_response(self, response_text: str, article_idx: int) -> List[Dict]:
        response = json.loads(self._smart_fix_json(response_text))
        samples = response.get("samples")
        if not isinstance(samples, list):
            raise ValueError("formulation JSON field 'samples' must be a list")
        article_title = response.get("article_title")
        article_doi = response.get("DOI")
        innovation = response.get("innovation")
        ai_summary = response.get("ai_summary")
        entity_rows = []
        part_rows = []

        for sample_idx, sample in enumerate(samples, 1):
            if not isinstance(sample, dict):
                raise ValueError(f"formulation sample {sample_idx} must be an object")
            sample["matrix_system"] = normalize_matrix_system_value(
                sample.get("matrix_system"),
                sample.get("matrix"),
                sample.get("EVA_polymer_fraction"),
                sample.get("PE_polymer_fraction"),
                sample.get("POE_polymer_fraction"),
            )
            sample_base = {
                "article_index": article_idx,
                "sample_index_in_article": sample_idx,
                "sample_name": sample.get("sample_name"),
                "article_title": article_title,
                "DOI": article_doi,
                "matrix": sample.get("matrix"),
                "matrix_system": sample.get("matrix_system"),
                "formulation_measurement_basis": sample.get("formulation_basis"),
                "EVA_fraction_in_matrix": sample.get("EVA_polymer_fraction"),
                "PE_fraction_in_matrix": sample.get("PE_polymer_fraction"),
                "POE_fraction_in_matrix": sample.get("POE_polymer_fraction"),
                "EVA_VA_content": sample.get("EVA_VA_content"),
                "PE_type": sample.get("PE_type"),
                "POE_type": sample.get("POE_type"),
                "matrix_key_data": sample.get("key_content"),
                "melt_flow_index": sample.get("melt_index"),
                "matrix_grade_information": sample.get("matrix_trname"),
                "total_additive_loading": sample.get("total_fr_loading"),
                "mixing_method": sample.get("mixing_method"),
                "formulation_notes": sample.get("formulation_notes"),
                "novelty": innovation,
                "ai_literature_summary": ai_summary,
            }

            entities = sample.get("entities") or []
            if not isinstance(entities, list):
                raise ValueError(f"formulation sample {sample_idx} field 'entities' must be a list")
            if not entities:
                entity_rows.append({col: clean_value(sample_base.get(col)) for col in ENTITY_COLUMNS})
                continue

            for entity_idx, entity in enumerate(entities, 1):
                if not isinstance(entity, dict):
                    raise ValueError(f"formulation entity {sample_idx}/{entity_idx} must be an object")
                entity_id = entity.get("entity_id") or f"E{entity_idx}"
                entity_name = entity.get("entity_name")
                row = dict(sample_base)
                row.update({
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "entity_role": entity.get("entity_role"),
                    "entity_type": entity.get("entity_type"),
                    "entity_loading": entity.get("entity_loading"),
                    "entity_loading_raw": entity.get("entity_loading_text"),
                    "entity_loading_unit": entity.get("entity_loading_unit"),
                    "entity_loading_basis": entity.get("entity_loading_basis"),
                    "entity_loading_wt_total_formulation": entity.get("entity_loading_wt_total"),
                    "entity_loading_phr_resin": entity.get("entity_loading_phr_resin"),
                    "entity_loading_conversion_notes": entity.get("entity_loading_conversion_note"),
                    "entity_material_type": entity.get("entity_material_type"),
                    "entity_grade": entity.get("entity_trade_name"),
                    "entity_profile": entity.get("entity_profile"),
                    "entity_figure_locator": entity.get("entity_fig_locator"),
                    "entity_bbox": entity.get("entity_bbox"),
                })
                entity_rows.append({col: clean_value(row.get(col)) for col in ENTITY_COLUMNS})

                parts = entity.get("parts") or []
                if not isinstance(parts, list):
                    raise ValueError(f"formulation entity {sample_idx}/{entity_idx} field 'parts' must be a list")
                for part_idx, part in enumerate(parts, 1):
                    if not isinstance(part, dict):
                        raise ValueError(f"formulation part {sample_idx}/{entity_idx}/{part_idx} must be an object")
                    part_row = {
                        "article_index": article_idx,
                        "sample_index_in_article": sample_idx,
                        "sample_name": sample.get("sample_name"),
                        "article_title": article_title,
                        "DOI": article_doi,
                        "entity_id": entity_id,
                        "entity_name": entity_name,
                        "part_name": part.get("part_name"),
                        "part_role": part.get("part_role"),
                        "part_relation": part.get("relation_to_entity"),
                        "modification_method": part.get("modification_method"),
                        "part_ratio": part.get("part_ratio"),
                        "part_loading": part.get("part_loading"),
                        "part_loading_unit": part.get("part_loading_unit"),
                        "part_figure_locator": part.get("part_fig_locator"),
                        "part_bbox": part.get("part_bbox"),
                        "part_smiles": part.get("part_smiles"),
                        "part_smiles_source": part.get("part_smiles_source"),
                        "part_notes": part.get("part_notes"),
                    }
                    part_rows.append({col: clean_value(part_row.get(col)) for col in PART_COLUMNS})

        self._set_pending_part_rows(part_rows)
        return entity_rows

    def append_to_excel(self, rows):
        append_rows_to_excel(self.output_excel, rows, self.column_order, dedupe_exact=True)
        append_rows_to_excel(self.parts_excel, self._get_pending_part_rows(), PART_COLUMNS, dedupe_exact=True)
        self._set_pending_part_rows([])

    def _drop_rows_for_file(self, file_id: str) -> int:
        count = super()._drop_rows_for_file(file_id)
        if os.path.exists(self.parts_excel):
            try:
                df = read_excel_canonical(self.parts_excel)
                if "source_id" in df.columns:
                    mask = df["source_id"].astype(str).eq(str(file_id))
                    part_count = int(mask.sum())
                    if part_count:
                        df.loc[~mask].to_excel(self.parts_excel, index=False)
                        count += part_count
            except Exception:
                pass
        return count

    def attach_source_metadata(self, rows: List[Dict], file_id: str, paper_dir: str) -> List[Dict]:
        rows = super().attach_source_metadata(rows, file_id, paper_dir)
        md_path = source_md_path(paper_dir)
        source_dir = os.path.normpath(os.path.abspath(paper_dir))
        for row in self._get_pending_part_rows():
            row["source_id"] = file_id
            row["source_folder"] = source_dir
            row["source_markdown_path"] = md_path
            attach_stable_identity(row, file_id)
        return rows


class PerformanceAnalyzer(SplitTaskAnalyzer):
    task_name = "performance"
    include_images = False

    def __init__(self, output_excel: str, model_name: str, input_folder: str):
        super().__init__(output_excel=output_excel, model_name=model_name, input_folder=input_folder)
        self.prompt_template = PERFORMANCE_PROMPT
        self.column_order = PERFORMANCE_COLUMNS
        self.formulation_excel = Path(self.output_dir) / "formulation_entities.xlsx"
        self._formulation_cache: Optional[pd.DataFrame] = None
        self._formulation_cache_lock = Lock()
        self._formulation_samples_cache: Dict[tuple, List[Dict]] = {}
        self._thread_context = threading.local()

    def _load_formulation_cache(self) -> pd.DataFrame:
        with self._formulation_cache_lock:
            if self._formulation_cache is not None:
                return self._formulation_cache
            if not self.formulation_excel.exists():
                self.log(f"[{self.task_name}] formulation results not found; sample-alignment context is unavailable: {self.formulation_excel}")
                self._formulation_cache = pd.DataFrame()
                return self._formulation_cache
            self._formulation_cache = read_excel_canonical(self.formulation_excel)
            self.log(f"[{self.task_name}] loaded formulation sample-alignment table: {self.formulation_excel}")
            return self._formulation_cache

    def _current_file_id(self) -> Optional[str]:
        return getattr(self._thread_context, "file_id", None)

    def _formulation_rows_for_paper(self, article_idx: Optional[int], file_id: Optional[str]) -> pd.DataFrame:
        df = self._load_formulation_cache()
        if df.empty:
            return pd.DataFrame()

        if file_id and "source_id" in df.columns:
            file_rows = df[df["source_id"].astype(str).str.strip().eq(str(file_id).strip())]
            if not file_rows.empty:
                return file_rows

        if article_idx is None or "article_index" not in df.columns:
            return pd.DataFrame()

        return df[df["article_index"].apply(lambda value: numeric_or_none(value) == article_idx)]

    def _formulation_samples_for_article(self, article_idx: Optional[int], file_id: Optional[str] = None) -> List[Dict]:
        cache_key = (
            str(file_id or "").strip(),
            normalize_key_number(article_idx),
        )
        with self._formulation_cache_lock:
            cached_samples = self._formulation_samples_cache.get(cache_key)
        if cached_samples is not None:
            return cached_samples

        article_rows = self._formulation_rows_for_paper(article_idx, file_id)
        if article_rows.empty:
            with self._formulation_cache_lock:
                self._formulation_samples_cache[cache_key] = []
            return []

        samples = []
        seen = set()
        for _, group in article_rows.groupby(["sample_index_in_article", "sample_name"], dropna=False, sort=True):
            sample_order = first_value(group, "sample_index_in_article")
            sample_name = first_value(group, "sample_name")
            sample_key = (sample_order, normalize_name(sample_name))
            if sample_key in seen:
                continue
            seen.add(sample_key)

            entity_summaries = []
            for _, entity in group.iterrows():
                role = normalize_role(entity.get("entity_role"))
                if role not in {"main_flame_retardant", "synergistic_flame_retardant", "matrix_polymer"}:
                    continue
                name = clean_value(entity.get("entity_name"))
                if is_missing(name):
                    continue
                loading = (
                    clean_value(entity.get("entity_loading_wt_total_formulation"))
                    or clean_value(entity.get("entity_loading_phr_resin"))
                    or clean_value(entity.get("entity_loading"))
                )
                if loading is not None:
                    entity_summaries.append(f"{name} ({role}, loading={loading})")
                else:
                    entity_summaries.append(f"{name} ({role})")

            samples.append({
                "article_sample_order": sample_order,
                "sample_name": sample_name,
                "matrix": first_value(group, "matrix"),
                "matrix_system": first_value(group, "matrix_system"),
                "total_fr_loading": first_value(group, "total_additive_loading"),
                "formulation_entities": "; ".join(dict.fromkeys(entity_summaries[:8])),
            })
        with self._formulation_cache_lock:
            self._formulation_samples_cache[cache_key] = samples
        return samples

    def _formulation_roster_text(self, article_idx: Optional[int], file_id: Optional[str]) -> Optional[str]:
        samples = self._formulation_samples_for_article(article_idx, file_id)
        if not samples:
            self.log(f"[{self.task_name}] no matching formulation sample roster; skipping formulation-dependent performance extraction: {file_id}")
            return None
        roster = {
            "article_idx": article_idx,
            "file_id": file_id,
            "instructions": [
                "Use these formulation samples as alignment anchors.",
                "When matched, output sample_name exactly as listed here.",
                "Return every roster sample; use # for metrics not reported for that sample.",
            ],
            "samples": samples,
        }
        return (
            "\n[FORMULATION SAMPLE ROSTER - ALIGNMENT ANCHORS]\n"
            + json.dumps(roster, ensure_ascii=False, indent=2)
            + "\n[END FORMULATION SAMPLE ROSTER]\n"
        )

    def build_content_blocks(
        self,
        markdown_text: str,
        image_paths: List[str],
        article_idx: Optional[int] = None,
        file_id: Optional[str] = None,
    ) -> List[Dict]:
        blocks = super().build_content_blocks(markdown_text, image_paths, article_idx=article_idx, file_id=file_id)
        roster_text = self._formulation_roster_text(article_idx, file_id)
        if roster_text:
            return [{"type": "text", "content": roster_text, "page": 0}] + blocks
        return blocks

    def _align_to_formulation_roster(self, article_idx: int, sample_name, sample_idx: int, file_id: Optional[str] = None) -> tuple:
        samples = self._formulation_samples_for_article(article_idx, file_id or self._current_file_id())
        if not samples:
            return sample_name, sample_idx, False
        normalized_target = normalize_name(sample_name)
        if not normalized_target:
            blank_candidates = [
                sample for sample in samples if not normalize_name(sample.get("sample_name"))
            ]
            if len(blank_candidates) != 1:
                return sample_name, sample_idx, False
            matched = blank_candidates[0]
            return matched.get("sample_name"), matched.get("article_sample_order") or sample_idx, True
        lookup = defaultdict(list)
        for sample in samples:
            normalized_name = normalize_name(sample.get("sample_name"))
            if normalized_name:
                lookup[normalized_name].append(sample)
        candidates = lookup.get(normalized_target, [])
        if len(candidates) > 1:
            exact_name = normalize_identity_text(sample_name)
            exact_candidates = [
                sample
                for sample in candidates
                if normalize_identity_text(sample.get("sample_name")) == exact_name
            ]
            candidates = exact_candidates
        if len(candidates) != 1:
            return sample_name, sample_idx, False
        matched = candidates[0]
        return matched.get("sample_name"), matched.get("article_sample_order") or sample_idx, True

    def parse_task_response(self, response_text: str, article_idx: int) -> List[Dict]:
        response = json.loads(self._smart_fix_json(response_text))
        article_title = response.get("article_title")
        article_doi = response.get("DOI")
        innovation = response.get("innovation")
        ai_summary = response.get("ai_summary")
        rows = []

        mapping = {
            "LOI": "LOI", "UL94_rating": "UL94_rating",
            "elongation_at_break": "elongation_at_break", "tensile_strength": "tensile_strength",
            "T1": "T1", "T5": "T5", "T10": "T10", "T50": "T50",
            "Tmax1": "Tmax1", "Tmax2": "Tmax2", "TGA_residue": "TGAresidue",
            "TTI": "TTI", "cone_heat_flux": "cone_heat_flux", "cone_specimen_thickness": "cone_sample_thickness",
            "pHRR": "pHRR", "THR": "THR", "pSPR": "pSPR", "TSP": "TSP",
            "pCOR": "pCOR", "pCO2R": "pCO2R", "cone_residue": "residue_rate",
            "performance_notes": "performance_notes",
        }

        response_samples = response.get("samples")
        if not isinstance(response_samples, list):
            raise ValueError("performance JSON field 'samples' must be a list")

        matched_roster_names = set()
        record_counts = defaultdict(int)
        for sample_idx, sample in enumerate(response_samples, 1):
            if not isinstance(sample, dict):
                raise ValueError(f"performance sample {sample_idx} must be an object")
            raw_sample_name = sample.get("sample_name")
            aligned_name, aligned_order, matched_roster = self._align_to_formulation_roster(
                article_idx,
                raw_sample_name,
                sample_idx,
                self._current_file_id(),
            )
            if matched_roster:
                matched_roster_names.add(normalize_name(aligned_name))
                if normalize_name(raw_sample_name) != normalize_name(aligned_name):
                    notes = clean_value(sample.get("performance_notes"))
                    alias_note = f"source performance label: {raw_sample_name}"
                    sample["performance_notes"] = f"{notes}; {alias_note}" if notes else alias_note
            row = {
                "article_index": article_idx,
                "sample_index_in_article": aligned_order,
                "sample_name": aligned_name,
                "article_title": article_title,
                "DOI": article_doi,
                "novelty": innovation,
                "ai_literature_summary": ai_summary,
                "formulation_alignment_status": "matched" if matched_roster else "unmatched",
            }
            for excel_col, json_key in mapping.items():
                row[excel_col] = sample.get(json_key)
            record_key = (normalize_key_number(aligned_order), normalize_name(aligned_name))
            record_counts[record_key] += 1
            row["performance_record_index"] = record_counts[record_key]
            row["performance_condition_signature"] = build_performance_condition_signature(row)
            rows.append({col: clean_value(row.get(col)) for col in PERFORMANCE_COLUMNS})

        for roster_sample in self._formulation_samples_for_article(article_idx, self._current_file_id()):
            roster_name = roster_sample.get("sample_name")
            roster_norm = normalize_name(roster_name)
            if not roster_norm or roster_norm in matched_roster_names:
                continue
            row = {
                "article_index": article_idx,
                "sample_index_in_article": roster_sample.get("article_sample_order"),
                "sample_name": roster_name,
                "article_title": article_title,
                "DOI": article_doi,
                "performance_notes": "No explicit performance value found for this formulation roster sample.",
                "novelty": innovation,
                "ai_literature_summary": ai_summary,
                "formulation_alignment_status": "roster_empty",
            }
            record_key = (
                normalize_key_number(roster_sample.get("article_sample_order")),
                normalize_name(roster_name),
            )
            record_counts[record_key] += 1
            row["performance_record_index"] = record_counts[record_key]
            row["performance_condition_signature"] = build_performance_condition_signature(row)
            rows.append({col: clean_value(row.get(col)) for col in PERFORMANCE_COLUMNS})
        return rows

    def process_paper_dir(self, paper_dir: str, article_idx: int = None):
        file_id = os.path.basename(os.path.normpath(paper_dir))
        if not self._formulation_samples_for_article(article_idx, file_id):
            self.log(f"[{self.task_name}] skipping article without matching formulation output: {file_id} (article_index {article_idx})")
            return "skipped"

        self._thread_context.file_id = file_id
        try:
            return super().process_paper_dir(paper_dir, article_idx=article_idx)
        finally:
            self._thread_context.file_id = None


def build_performance_lookups(performance: pd.DataFrame) -> tuple:
    exact_lookup = {}
    name_lookup = {}

    for _, row in performance.iterrows():
        sample_norm = normalize_name(row.get("sample_name"))
        if not sample_norm:
            continue
        article_key = normalize_key_number(row.get("article_index"))
        order_key = normalize_key_number(row.get("sample_index_in_article"))
        exact_lookup.setdefault((article_key, order_key, sample_norm), []).append(row)
        name_lookup.setdefault((article_key, sample_norm), []).append(row)

    return exact_lookup, name_lookup


def lookup_performance_row(exact_lookup: Dict, name_lookup: Dict, article_no, sample_order, sample_name):
    sample_norm = normalize_name(sample_name)
    if not sample_norm:
        return None

    article_key = normalize_key_number(article_no)
    order_key = normalize_key_number(sample_order)
    exact_rows = exact_lookup.get((article_key, order_key, sample_norm), [])
    if len(exact_rows) == 1:
        return exact_rows[0]

    name_rows = name_lookup.get((article_key, sample_norm), [])
    if len(name_rows) == 1:
        return name_rows[0]

    return None


def select_entity(rows: pd.DataFrame, roles: set, excluded_names: Optional[set] = None) -> Optional[pd.Series]:
    excluded_names = excluded_names or set()
    candidates = []
    for _, row in rows.iterrows():
        role = normalize_role(row.get("entity_role"))
        name_norm = normalize_name(row.get("entity_name"))
        if role in roles and name_norm not in excluded_names and name_norm:
            candidates.append(row)
    if not candidates:
        return None
    return sorted(candidates, key=loading_value_for_sort, reverse=True)[0]


def parts_for_entity(parts: pd.DataFrame, entity_row: Optional[pd.Series]) -> pd.DataFrame:
    if entity_row is None or parts.empty:
        return pd.DataFrame(columns=PART_COLUMNS)

    same_sample = parts[
        (parts["article_index"] == entity_row.get("article_index"))
        & (parts["sample_index_in_article"] == entity_row.get("sample_index_in_article"))
    ]
    if same_sample.empty:
        return pd.DataFrame(columns=PART_COLUMNS)

    entity_id = normalize_name(entity_row.get("entity_id"))
    entity_name = normalize_name(entity_row.get("entity_name"))
    if entity_id:
        matched = same_sample[same_sample["entity_id"].apply(normalize_name) == entity_id]
        if not matched.empty:
            return matched
    if entity_name:
        matched = same_sample[same_sample["entity_name"].apply(normalize_name) == entity_name]
        if not matched.empty:
            return matched
    return pd.DataFrame(columns=PART_COLUMNS)


def entity_is_modified(entity_row: Optional[pd.Series], part_rows: pd.DataFrame) -> bool:
    if entity_row is None:
        return False
    entity_type = normalize_role(entity_row.get("entity_type"))
    modified_types = {
        "surface_treated", "grafted_additive", "core_shell", "microencapsulated",
        "coated", "intercalated", "doped", "in_situ_grown", "hybrid_complex",
        "matrix_grafted", "blend_package",
    }
    if entity_type in modified_types:
        return True
    if part_rows.empty:
        return False
    methods = [normalize_role(v) for v in part_rows.get("modification_method", [])]
    return any(method not in {"", "none", "unknown"} for method in methods)


def entity_part_summary(part_rows: pd.DataFrame) -> Dict:
    if part_rows.empty:
        return {
            "base_names": None,
            "base_smiles": "#",
            "base_smiles_source": None,
            "modifier_names": None,
            "modifier_smiles": "#",
            "modifier_smiles_source": None,
            "methods": None,
            "ratios": None,
            "loadings": None,
            "locator": None,
            "bbox": None,
            "smiles": "#",
            "smiles_source": None,
        }

    base_roles = {"base", "core", "grafted_backbone"}
    modifier_roles = {
        "shell", "coating_layer", "grafted_moiety", "dopant", "intercalated_guest",
        "surface_ligand", "coupling_agent", "complexing_agent", "synergistic_part",
        "modifier",
    }
    base_parts = []
    modifier_parts = []
    for _, part in part_rows.iterrows():
        role = normalize_role(part.get("part_role"))
        if role in base_roles:
            base_parts.append(part)
        elif role in modifier_roles:
            modifier_parts.append(part)

    if not modifier_parts:
        base_norms = {normalize_name(p.get("part_name")) for p in base_parts}
        modifier_parts = [
            part for _, part in part_rows.iterrows()
            if normalize_name(part.get("part_name")) not in base_norms
        ]

    modifier_df = pd.DataFrame(modifier_parts)
    base_df = pd.DataFrame(base_parts)
    target_df = modifier_df if not modifier_df.empty else part_rows
    methods = [
        v for v in target_df.get("modification_method", pd.Series(dtype=object)).tolist()
        if not is_missing(v) and normalize_role(v) != "none"
    ]
    locator = first_value(target_df, "part_figure_locator")
    bbox = first_value(target_df, "part_bbox")
    smiles = joined_column_values(target_df, "part_smiles") or "#"
    smiles_source = joined_column_values(target_df, "part_smiles_source")

    return {
        "base_names": join_unique([p.get("part_name") for p in base_parts]),
        "base_smiles": joined_column_values(base_df, "part_smiles") or "#",
        "base_smiles_source": joined_column_values(base_df, "part_smiles_source"),
        "modifier_names": join_unique([p.get("part_name") for p in modifier_parts]),
        "modifier_smiles": joined_column_values(modifier_df, "part_smiles") or "#",
        "modifier_smiles_source": joined_column_values(modifier_df, "part_smiles_source"),
        "methods": join_unique(methods),
        "ratios": join_unique(target_df.get("part_ratio", pd.Series(dtype=object)).tolist()),
        "loadings": join_unique(target_df.get("part_loading", pd.Series(dtype=object)).tolist()),
        "locator": locator,
        "bbox": bbox,
        "smiles": smiles,
        "smiles_source": smiles_source,
    }


def entity_to_fr(row: Optional[pd.Series], prefix: str) -> Dict:
    if row is None:
        return {
            f"{prefix}": None,
            f"{prefix}_loading_phr": None,
            f"{prefix}_loading_wt": None,
            f"{prefix}_type": None,
            f"{prefix}_grade": None,
            f"{prefix}_profile": None,
            f"{prefix}_figure_locator": None,
            f"{prefix}_bbox": None,
            f"{prefix}_smiles": "#",
            f"{prefix}_smiles_source": None,
        }

    locator = row.get("entity_figure_locator")
    name = row.get("entity_name")
    wt_total = clean_value(row.get("entity_loading_wt_total_formulation"))
    phr = clean_value(row.get("entity_loading_phr_resin"))
    smiles = clean_value(row.get("entity_smiles")) or "#"
    smiles_source = clean_value(row.get("entity_smiles_source"))
    return {
        f"{prefix}": clean_value(name),
        f"{prefix}_loading_phr": phr,
        f"{prefix}_loading_wt": wt_total,
        f"{prefix}_type": clean_value(row.get("entity_material_type")),
        f"{prefix}_grade": clean_value(row.get("entity_grade")),
        f"{prefix}_profile": clean_value(row.get("entity_profile")),
        f"{prefix}_figure_locator": clean_value(locator),
        f"{prefix}_bbox": clean_value(row.get("entity_bbox")),
        f"{prefix}_smiles": smiles,
        f"{prefix}_smiles_source": smiles_source,
    }


def rows_for_roles(rows: pd.DataFrame, roles: set, keywords: Optional[List[str]] = None) -> pd.DataFrame:
    if rows.empty:
        return rows
    role_mask = rows["entity_role"].apply(normalize_role).isin(roles)
    if not keywords:
        return rows[role_mask]

    keyword_mask = pd.Series(False, index=rows.index)
    keyword_columns = ["entity_name", "entity_type", "entity_grade", "entity_profile"]
    lowered_keywords = [keyword.lower() for keyword in keywords]
    for column in keyword_columns:
        if column not in rows.columns:
            continue
        text = rows[column].fillna("").astype(str).str.lower()
        column_mask = pd.Series(False, index=rows.index)
        for keyword in lowered_keywords:
            column_mask = column_mask | text.str.contains(re.escape(keyword), na=False)
        keyword_mask = keyword_mask | column_mask
    return rows[role_mask | keyword_mask]


def entity_collection_summary(rows: pd.DataFrame) -> Dict:
    if rows.empty:
        return {
            "names": None,
            "loadings": None,
            "wt_total": None,
            "phr": None,
            "type": None,
            "trade": None,
            "locator": None,
            "bbox": None,
            "smiles": "#",
            "smiles_source": None,
        }

    locator = first_value(rows, "entity_figure_locator")
    smiles = joined_column_values(rows, "entity_smiles") or "#"
    smiles_source = joined_column_values(rows, "entity_smiles_source")
    return {
        "names": join_unique(rows.get("entity_name", pd.Series(dtype=object)).tolist()),
        "loadings": join_unique(rows.get("entity_loading", pd.Series(dtype=object)).tolist()),
        "wt_total": sum_numeric(rows.get("entity_loading_wt_total_formulation", pd.Series(dtype=object)).tolist()),
        "phr": sum_numeric(rows.get("entity_loading_phr_resin", pd.Series(dtype=object)).tolist()),
        "type": first_value(rows, "entity_type"),
        "trade": join_unique(rows.get("entity_grade", pd.Series(dtype=object)).tolist()),
        "locator": locator,
        "bbox": first_value(rows, "entity_bbox"),
        "smiles": smiles,
        "smiles_source": smiles_source,
    }


def add_entity_collection(
    row: Dict,
    group: pd.DataFrame,
    prefix: str,
    roles: set,
    keywords: Optional[List[str]] = None,
) -> pd.DataFrame:
    matched = rows_for_roles(group, roles, keywords=keywords)
    info = entity_collection_summary(matched)
    row[prefix] = info["names"]
    row[f"{prefix}_loading_phr"] = info["phr"]
    row[f"{prefix}_loading_wt"] = info["wt_total"]
    for suffix, key in [
        ("_type", "type"),
        ("_grade", "trade"),
        ("_figure_locator", "locator"),
        ("_bbox", "bbox"),
        ("_smiles", "smiles"),
        ("_smiles_source", "smiles_source"),
    ]:
        column = f"{prefix}{suffix}"
        if column in MERGED_COLUMNS:
            row[column] = info[key]
    return matched


def sum_numeric(values) -> Optional[float]:
    total = 0.0
    found = False
    for value in values:
        number = numeric_or_none(value)
        if number is None:
            continue
        total += number
        found = True
    if not found:
        return None
    return round(total, 6)


def merge_entity_outputs(
    entities_excel: Path,
    parts_excel: Path,
    performance_excel: Path,
    merged_excel: Path,
    release_safe: bool = False,
):
    if not entities_excel.exists():
        raise FileNotFoundError(f"Missing formulation entity table: {entities_excel}")
    if not performance_excel.exists():
        raise FileNotFoundError(f"Missing performance table: {performance_excel}")

    entities = read_excel_canonical(entities_excel)
    parts = read_excel_canonical(parts_excel) if parts_excel.exists() else pd.DataFrame(columns=PART_COLUMNS)
    performance = read_excel_canonical(performance_excel)
    rows = []
    output_columns = RELEASE_MERGED_COLUMNS if release_safe else MERGED_COLUMNS
    sample_id = 1

    perf_exact_lookup, perf_name_lookup = build_performance_lookups(performance)

    group_cols = ["article_index", "sample_index_in_article", "sample_name"]
    for _, group in entities.groupby(group_cols, dropna=False, sort=True):
        article_no = first_value(group, "article_index")
        sample_order = first_value(group, "sample_index_in_article")
        sample_name = first_value(group, "sample_name")
        perf = lookup_performance_row(perf_exact_lookup, perf_name_lookup, article_no, sample_order, sample_name)
        if perf is None and is_missing(sample_name):
            same_order = performance[
                (performance["article_index"] == article_no)
                & (performance["sample_index_in_article"] == sample_order)
            ]
            perf = same_order.iloc[0] if not same_order.empty else None

        group = compute_loading_conversions(group)
        fr1 = select_entity(group, {"main_flame_retardant"})
        excluded = {normalize_name(fr1.get("entity_name"))} if fr1 is not None else set()
        fr2 = select_entity(group, {"synergistic_flame_retardant"}, excluded_names=excluded)
        if fr2 is None:
            fr2 = select_entity(group, {"main_flame_retardant"}, excluded_names=excluded)

        row = {
            "sample_number": sample_id,
            "article_index": article_no,
            "sample_index_in_article": sample_order,
            "sample_name": sample_name,
            "article_title": first_value(group, "article_title") or (perf.get("article_title") if perf is not None else None),
            "DOI": first_value(group, "DOI") or (perf.get("DOI") if perf is not None else None),
            "source_id": first_value(group, "source_id"),
            "source_folder": first_value(group, "source_folder"),
            "source_markdown_path": first_value(group, "source_markdown_path"),
            "article_uid": first_value(group, "article_uid") or first_value(group, "source_id"),
            "sample_uid": first_value(group, "sample_uid"),
            "matrix": first_value(group, "matrix") or first_value(group, "matrix_system"),
            "matrix_system": first_value(group, "matrix_system"),
            "formulation_measurement_basis": first_value(group, "formulation_measurement_basis"),
            "EVA_fraction_in_matrix": first_value(group, "EVA_fraction_in_matrix"),
            "PE_fraction_in_matrix": first_value(group, "PE_fraction_in_matrix"),
            "POE_fraction_in_matrix": first_value(group, "POE_fraction_in_matrix"),
            "EVA_VA_content": first_value(group, "EVA_VA_content"),
            "PE_type": first_value(group, "PE_type"),
            "POE_type": first_value(group, "POE_type"),
            "matrix_key_data": first_value(group, "matrix_key_data"),
            "melt_flow_index": first_value(group, "melt_flow_index"),
            "intrinsic_type_indicator": 0,
            "matrix_grade_information": first_value(group, "matrix_grade_information"),
            "total_additive_loading": first_value(group, "total_additive_loading"),
            "mixing_method": first_value(group, "mixing_method"),
            "formulation_notes": first_value(group, "formulation_notes"),
            "novelty": first_value(group, "novelty") or (perf.get("novelty") if perf is not None else None),
            "ai_literature_summary": first_value(group, "ai_literature_summary") or (perf.get("ai_literature_summary") if perf is not None else None),
        }
        row.update(entity_to_fr(fr1, "flame_retardant_1"))
        row.update(entity_to_fr(fr2, "flame_retardant_2"))

        for idx, fr in [(1, fr1), (2, fr2)]:
            prefix = f"flame_retardant_{idx}"
            part_rows = parts_for_entity(parts, fr)
            part_info = entity_part_summary(part_rows)
            modified = entity_is_modified(fr, part_rows)
            entity_type = clean_value(fr.get("entity_type")) if fr is not None else None
            entity_name = clean_value(fr.get("entity_name")) if fr is not None else None

            row[f"{prefix}_entity_type"] = entity_type
            row[f"{prefix}_base_or_core_components"] = part_info["base_names"]
            row[f"{prefix}_base_or_core_components_smiles"] = part_info["base_smiles"]
            row[f"{prefix}_base_or_core_components_smiles_source"] = part_info["base_smiles_source"]
            row[f"{prefix}_shell_grafted_or_modified_components"] = part_info["modifier_names"]
            row[f"{prefix}_shell_grafted_or_modified_components_smiles"] = part_info["modifier_smiles"]
            row[f"{prefix}_shell_grafted_or_modified_components_smiles_source"] = part_info["modifier_smiles_source"]
            row[f"{prefix}_component_ratio"] = part_info["ratios"]
            row[f"{prefix}_is_modified"] = 1 if modified else 0
            row[f"{prefix}_modified_product_name"] = entity_name if modified else None
            row[f"{prefix}_modifier"] = part_info["modifier_names"]
            row[f"{prefix}_modification_method"] = part_info["methods"] or (entity_type if modified else None)
            row[f"{prefix}_modification_ratio"] = part_info["ratios"]
            row[f"{prefix}_modifier_loading"] = part_info["loadings"]
            row[f"{prefix}_modifier_figure_locator"] = part_info["locator"]
            row[f"{prefix}_modifier_bbox"] = part_info["bbox"]
            row[f"{prefix}_modifier_smiles"] = part_info["modifier_smiles"] if part_info["modifier_smiles"] != "#" else part_info["smiles"]
            row[f"{prefix}_modifier_smiles_source"] = part_info["modifier_smiles_source"] or part_info["smiles_source"]

        selected_names = {normalize_name(fr.get("entity_name")) for fr in [fr1, fr2] if fr is not None}
        extra_fr = []
        for _, entity in group.iterrows():
            role = normalize_role(entity.get("entity_role"))
            name = entity.get("entity_name")
            if role in {"main_flame_retardant", "synergistic_flame_retardant"} and normalize_name(name) not in selected_names:
                extra_fr.append(str(name))
        row["other_flame_retardant_or_synergist_components"] = "; ".join(dict.fromkeys(extra_fr)) or None

        main_fr_rows = rows_for_roles(group, {"main_flame_retardant"})
        synergist_rows = rows_for_roles(group, {"synergistic_flame_retardant"})
        functional_rows = add_entity_collection(row, group, "functional_fillers", {"functional_filler"})
        compat_rows = add_entity_collection(row, group, "compatibilizers", {"compatibilizer"})
        crosslink_rows = add_entity_collection(row, group, "crosslinking_agents", {"crosslinking_agent"})
        co_crosslink_rows = add_entity_collection(row, group, "co_crosslinking_agents", {"co_crosslinking_agent"})
        processing_rows = add_entity_collection(row, group, "processing_aids", {"processing_aid"})
        antioxidant_rows = add_entity_collection(
            row,
            group,
            "antioxidants_or_stabilizers",
            {"antioxidant", "stabilizer", "antioxidant_stabilizer", "metal_deactivator"},
            keywords=[
                "antioxidant",
                "anti-oxidant",
                "stabilizer",
                "stabiliser",
                "hindered phenol",
                "phosphite",
                "irganox",
                "irgafos",
                "antioxidant 1010",
                "antioxidant1010",
                "antioxidant 1076",
                "antioxidant1076",
                "ao-1010",
                "ao-1076",
                "ao-168",
                "pilnox tdq",
                "tdq",
            ],
        )
        other_group = group.drop(index=antioxidant_rows.index, errors="ignore")
        other_additive_rows = add_entity_collection(row, other_group, "other_additives", {"other_additive"})

        synergist_info = entity_collection_summary(synergist_rows)
        row["synergistic_flame_retardants"] = synergist_info["names"]
        row["synergistic_flame_retardants_loading_phr"] = synergist_info["phr"]
        row["synergistic_flame_retardants_loading_wt"] = synergist_info["wt_total"]
        row["contains_compatibilizer"] = 0 if compat_rows.empty else 1
        row["is_crosslinked"] = 0 if crosslink_rows.empty and co_crosslink_rows.empty else 1
        row["contains_processing_aid"] = 0 if processing_rows.empty else 1
        row["contains_antioxidant"] = 0 if antioxidant_rows.empty else 1
        row["main_flame_retardant_total_wt"] = sum_numeric(main_fr_rows.get("entity_loading_wt_total_formulation", pd.Series(dtype=object)).tolist())
        row["synergistic_flame_retardant_total_wt"] = synergist_info["wt_total"]
        row["functional_filler_total_wt"] = sum_numeric(functional_rows.get("entity_loading_wt_total_formulation", pd.Series(dtype=object)).tolist())
        row["compatibilizer_total_wt"] = sum_numeric(compat_rows.get("entity_loading_wt_total_formulation", pd.Series(dtype=object)).tolist())
        row["crosslinking_agent_total_wt"] = sum_numeric(
            pd.concat([crosslink_rows, co_crosslink_rows], ignore_index=True).get(
                "entity_loading_wt_total_formulation", pd.Series(dtype=object)
            ).tolist()
        )
        row["processing_aid_total_wt"] = sum_numeric(processing_rows.get("entity_loading_wt_total_formulation", pd.Series(dtype=object)).tolist())
        row["antioxidant_or_stabilizer_total_wt"] = sum_numeric(antioxidant_rows.get("entity_loading_wt_total_formulation", pd.Series(dtype=object)).tolist())

        if perf is not None:
            for col in PERFORMANCE_COLUMNS:
                if col in {
                    "article_index", "sample_index_in_article", "sample_name", "article_title", "DOI",
                    *SOURCE_COLUMNS, *IDENTITY_COLUMNS, *PERFORMANCE_IDENTITY_COLUMNS,
                    "novelty", "ai_literature_summary",
                }:
                    continue
                row[col] = perf.get(col)

        rows.append({col: clean_value(row.get(col)) for col in output_columns})
        sample_id += 1

    merged = pd.DataFrame(rows, columns=output_columns)
    merged_excel.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(merged_excel, index=False)
    print(f"Entity merge completed: {merged_excel}")
    print(f"Samples: {len(merged)}")


def load_api_keys(path: Optional[str]) -> List[str]:
    if not path:
        return []
    key_path = Path(path)
    if not key_path.exists():
        raise FileNotFoundError(f"API keys file not found: {key_path}")
    keys = []
    for line in key_path.read_text(encoding="utf-8").splitlines():
        value = line.strip().lstrip("\ufeff")
        if value and not value.startswith("#"):
            keys.append(value)
    if not keys:
        raise ValueError(f"No API keys found in: {key_path}")
    return keys


def make_api_key_queue(keys: List[str]) -> Optional[Queue]:
    if not keys:
        return None
    queue = Queue(maxsize=len(keys))
    for key in keys:
        queue.put(key)
    return queue


def configure_runtime(analyzer, args, api_key_queue: Optional[Queue]):
    analyzer.dry_run = args.dry_run
    analyzer.force = args.force
    analyzer.api_url = args.api_url
    analyzer.limit = args.limit
    analyzer.workers = args.workers
    analyzer.api_retries = args.api_retries
    analyzer.retry_base_delay = args.retry_base_delay
    analyzer.rate_limit_sleep = args.rate_limit_sleep
    analyzer.rate_limit_retries = args.rate_limit_retries
    analyzer.read_timeout = args.read_timeout
    analyzer.max_tokens = args.max_tokens
    analyzer.stream = args.stream
    analyzer._api_key_queue = api_key_queue
    if args.force_vision or str(args.model).lower().startswith("gpt-5"):
        analyzer.supports_vision = True
        analyzer.log("Vision-format requests enabled.")


def run_formulation(args):
    analyzer = EntityFormulationAnalyzer(
        output_excel=str(Path(args.output_dir) / "formulation_entities.xlsx"),
        parts_excel=str(Path(args.output_dir) / "formulation_entity_parts.xlsx"),
        model_name=args.model,
        input_folder=args.input,
    )
    configure_runtime(analyzer, args, make_api_key_queue(load_api_keys(args.api_keys_file)))
    analyzer.process_folder(args.input)
    analyzer.summarize_run()


def run_performance(args):
    analyzer = PerformanceAnalyzer(
        output_excel=str(Path(args.output_dir) / "performance_samples.xlsx"),
        model_name=args.model,
        input_folder=args.input,
    )
    configure_runtime(analyzer, args, make_api_key_queue(load_api_keys(args.api_keys_file)))
    analyzer.process_folder(args.input)
    analyzer.summarize_run()


VISUAL_PERFORMANCE_PROMPT = """
You are a supplemental figure-context and image performance extractor for an EVA/PE/POE flame-retardant performance database.

If one sample has values under multiple test conditions, emit one separate samples-array record per condition. Repeat the exact sample_name, preserve every condition-specific record, and state the distinguishing condition/image panel in performance_notes. Never collapse records merely because sample_name is the same.

This task extracts supplemental sample-level performance values from the supplied figure-context blocks and images.
Each image may be preceded by a short figure-context block from the markdown file. You may use that local context as text evidence when it directly states the same performance metric, sample, value, and unit.
Do NOT use prior text-performance output as performance-value evidence.
The formulation roster may be used ONLY to align sample names; it is not performance evidence.
Return valid JSON only. Do not output Markdown, explanations, or extra text.

Use "#" for missing values.
Every extracted structured performance metric MUST follow this source marking rule:
- Use no prefix for values directly stated in the supplied figure-context text / nearby markdown text.
- Use "w-" for image-word values: printed numbers/words inside an image, including figure labels, point labels, bar labels, table cells rendered as an image, or numeric annotations printed on a panel.
- Use "l-" for image-line values: values estimated from plotted line/curve position, bar height without a printed numeric label, scatter point location, axis scale, or curve endpoint.
If a value comes from image evidence and has no "w-" or "l-" prefix, it is invalid. If it comes from figure-context text, do not add a prefix.

[VISUAL PRIORITY AND LIMITS]
1. Extract a value only when the metric name, unit, sample identity, and evidence source are clear from the figure context or image evidence.
2. If image evidence is ambiguous, crowded, low-resolution, unlabeled, or cannot be aligned to a roster sample, output "#".
3. Do NOT infer values from trends, mechanism schemes, SEM/TEM photos, residue photos, or text descriptions outside the image.
4. Do NOT average image values with text values. If local context text and image-line estimates conflict, prefer the explicit text/printed numeric value and explain the conflict in performance_notes.
5. For UL94_rating, encode V-0/V-1/V-2/NR/HB/no rating as 0/1/2/3. Use plain 0/1/2/3 for explicit figure-context text, and use "w-0", "w-1", "w-2", or "w-3" when the rating label is visibly printed in the image. Never use "l-" for UL-94.
6. If a figure shows a text table as an image, table-cell values are "w-" values.
7. If a figure prints a numeric annotation next to a curve point, that value is "w-" even though it is on a curve.
8. If a TGA/DTG plot prints T50/Tmax/residue annotations, those values are "w-" values. Only unprinted coordinate reads are "l-" values.
9. If a cone-calorimetry curve gives only THE/total heat in kJ and the image does not show exposed area, output "#" for THR. Do not perform area normalization unless the image itself provides area or the value is already in MJ/m2.
10. Mention the image filename/panel and whether the value is context-text, image-word, or image-line in performance_notes.
11. If a value is present in figure-context text but not visible inside the image itself, you may output it without prefix when the metric, sample, value, and unit are explicit.

[METRIC DEFINITIONS]
LOI: limiting oxygen index, numeric % or vol%.
UL94_rating: V-0=0, V-1=1, V-2=2, NR/HB/no rating=3.
elongation_at_break: tensile elongation at break, numeric %.
tensile_strength: tensile strength/stress, numeric MPa.
T1/T5/T10/T50: TGA weight-loss temperatures, numeric deg C.
Tmax1/Tmax2: DTG peak temperatures, numeric deg C.
TGAresidue: TGA residue/char yield, numeric %.
TTI: cone-calorimeter time to ignition only, numeric s.
cone_heat_flux: cone/fire heat flux, numeric kW/m2.
cone_sample_thickness: cone/fire specimen thickness, numeric mm.
pHRR: peak heat-release rate, numeric kW/m2.
THR: total heat release, numeric MJ/m2.
pSPR: peak smoke production rate, numeric m2/s.
TSP: total smoke production/release, numeric m2.
pCOR: peak CO production rate, numeric when explicitly reported.
pCO2R: peak CO2 production rate, numeric when explicitly reported.
residue_rate: cone/fire residue rate, numeric %.

[OUTPUT JSON SCHEMA]
{
  "article_title": "# if not visible in image",
  "DOI": "# if not visible in image",
  "innovation": "#",
  "ai_summary": "Supplemental figure-context/image performance extraction.",
  "samples": [
    {
      "sample_name": "exact roster/sample label",
      "LOI": "# or plain/w-/l- value",
      "UL94_rating": "# or plain 0/1/2/3 or w-0/w-1/w-2/w-3",
      "elongation_at_break": "# or plain/w-/l- value",
      "tensile_strength": "# or plain/w-/l- value",
      "T1": "# or plain/w-/l- value",
      "T5": "# or plain/w-/l- value",
      "T10": "# or plain/w-/l- value",
      "T50": "# or plain/w-/l- value",
      "Tmax1": "# or plain/w-/l- value",
      "Tmax2": "# or plain/w-/l- value",
      "TGAresidue": "# or plain/w-/l- value",
      "TTI": "# or plain/w-/l- value",
      "cone_heat_flux": "# or plain/w-/l- value",
      "cone_sample_thickness": "# or plain/w-/l- value",
      "pHRR": "# or plain/w-/l- value",
      "THR": "# or plain/w-/l- value",
      "pSPR": "# or plain/w-/l- value",
      "TSP": "# or plain/w-/l- value",
      "pCOR": "# or plain/w-/l- value",
      "pCO2R": "# or plain/w-/l- value",
      "residue_rate": "# or plain/w-/l- value",
      "performance_notes": "image filename/panel and source type"
    }
  ]
}
"""

COL_ARTICLE = "article_index"
COL_SAMPLE_ORDER = "sample_index_in_article"
COL_SAMPLE_NAME = "sample_name"
COL_NOTES = "performance_notes"

METRIC_COLUMNS = [
    "LOI", "UL94_rating", "elongation_at_break", "tensile_strength",
    "T1", "T5", "T10", "T50", "Tmax1", "Tmax2", "TGA_residue",
    "TTI", "cone_heat_flux", "cone_specimen_thickness",
    "pHRR", "THR", "pSPR", "TSP", "pCOR", "pCO2R", "cone_residue",
]

KEY_COLUMNS = [COL_ARTICLE, COL_SAMPLE_ORDER, COL_SAMPLE_NAME]


class VisualPerformanceAnalyzer(PerformanceAnalyzer):
    task_name = "visual_performance"
    include_images = True

    def __init__(self, output_excel: str, model_name: str, input_folder: str):
        super().__init__(output_excel=output_excel, model_name=model_name, input_folder=input_folder)
        self.prompt_template = VISUAL_PERFORMANCE_PROMPT
        self.temperature = 0.0

    def build_content_blocks(
        self,
        markdown_text: str,
        image_paths: List[str],
        article_idx: Optional[int] = None,
        file_id: Optional[str] = None,
    ) -> List[Dict]:
        blocks: List[Dict] = []
        roster_text = self._formulation_roster_text(article_idx, file_id)
        if roster_text:
            blocks.append({"type": "text", "content": roster_text, "page": 0})
        blocks.append({
            "type": "text",
            "content": (
                "[SUPPLEMENTAL FIGURE-CONTEXT AND IMAGE TASK]\n"
                "Use the supplied figure-context blocks and images below as supplemental performance evidence. "
                "Mark context-text values with no prefix, image-word values with w-, and image-line/curve-estimated values with l-. "
                "The roster above is only for sample-name alignment.\n"
            ),
            "page": 0,
        })
        image_contexts = extract_image_contexts_from_markdown(markdown_text, image_paths)
        for image_path in image_paths:
            try:
                image_id = os.path.basename(image_path)
                context = image_contexts.get(image_id)
                if context:
                    blocks.append({
                        "type": "text",
                        "content": (
                            f"[FIGURE CONTEXT FOR IMAGE ID: {image_id}]\n"
                            f"{context}\n"
                            "[END FIGURE CONTEXT]\n"
                        ),
                        "page": 0,
                    })
                with open(image_path, "rb") as f:
                    b64_image = base64.b64encode(f.read()).decode("utf-8")
                blocks.append({
                    "type": "image",
                    "content": b64_image,
                    "page": 0,
                    "image_path": image_path,
                    "image_id": image_id,
                    "category": "image",
                })
            except Exception as exc:
                self.log(f"Failed to read image {image_path}: {exc}")
        return blocks


def has_visual_tag(value) -> bool:
    text = clean_value(value)
    return isinstance(text, str) and (text.startswith("w-") or text.startswith("l-"))


def has_supplement_value(value) -> bool:
    text = clean_value(value)
    if text is None:
        return False
    return not value_missing(text)


def normalize_metric_output_value(value):
    text = clean_value(value)
    if not isinstance(text, str):
        return text

    prefix = ""
    body = text.strip()
    if body.startswith("w-") or body.startswith("l-"):
        prefix = body[:2]
        body = body[2:].strip()

    body = body.replace("\u00c2\u00b1", "\u00b1").replace("+/-", "\u00b1").replace("+-", "\u00b1")
    body = re.split('\\s*\xb1\\s*', body, maxsplit=1)[0].strip()
    body = re.sub(r"\s*(?:%|vol%|MPa|s|sec|\u00b0C|deg C|kW/m2|kW/m\u00b2|MJ/m2|MJ/m\u00b2|m2/s|m\u00b2/s|m2|m\u00b2)\s*$", "", body).strip()

    return prefix + body if prefix else body


def value_missing(value) -> bool:
    return is_missing(value) or str(value).strip().lower() in {"nan", "none", "null"}


def row_key(row: pd.Series):
    return (
        normalize_key_number(row.get(COL_ARTICLE)),
        normalize_key_number(row.get(COL_SAMPLE_ORDER)),
        normalize_name(row.get(COL_SAMPLE_NAME)),
    )


def performance_sample_key(row: pd.Series) -> tuple:
    return ("paper", *row_key(row))


def performance_match_key(row: pd.Series) -> tuple:
    condition_signature = clean_value(row.get("performance_condition_signature"))
    if condition_signature is None:
        condition_signature = build_performance_condition_signature(row)
    return (*performance_sample_key(row), str(condition_signature or "default"))


def build_visual_lookup(visual_df: pd.DataFrame) -> Dict[tuple, deque]:
    lookup: Dict[tuple, deque] = defaultdict(deque)
    if visual_df.empty:
        return lookup
    for row_index, row in visual_df.iterrows():
        key = performance_match_key(row)
        sample_key = performance_sample_key(row)
        if not sample_key[-1]:
            continue
        lookup[key].append((row_index, row))
    return lookup


def performance_condition_match_strength(primary_row: pd.Series, visual_row: pd.Series) -> int:
    shared_conditions = 0
    for column in PERFORMANCE_CONDITION_COLUMNS:
        primary_value = clean_value(primary_row.get(column))
        visual_value = clean_value(visual_row.get(column))
        if primary_value is None or visual_value is None:
            continue
        if normalize_performance_condition_value(primary_value) != normalize_performance_condition_value(visual_value):
            return -1
        shared_conditions += 1
    return shared_conditions


def performance_conditions_compatible(primary_row: pd.Series, visual_row: pd.Series) -> bool:
    return performance_condition_match_strength(primary_row, visual_row) > 0


def pair_visual_performance_rows(primary: pd.DataFrame, visual: pd.DataFrame) -> Dict:
    exact_lookup = build_visual_lookup(visual)
    pairings = {}
    used_visual_indices = set()

    for primary_index, primary_row in primary.iterrows():
        candidates = exact_lookup.get(performance_match_key(primary_row))
        if not candidates:
            continue
        while candidates and candidates[0][0] in used_visual_indices:
            candidates.popleft()
        if candidates:
            visual_index, visual_row = candidates.popleft()
            pairings[primary_index] = (visual_index, visual_row)
            used_visual_indices.add(visual_index)

    while True:
        visual_candidates = {}
        primary_candidate_counts = defaultdict(int)
        unmatched_primary = [index for index in primary.index if index not in pairings]
        primary_by_sample = defaultdict(list)
        for primary_index in unmatched_primary:
            primary_by_sample[performance_sample_key(primary.loc[primary_index])].append(primary_index)

        for visual_index, visual_row in visual.iterrows():
            if visual_index in used_visual_indices:
                continue
            sample_primary = primary_by_sample.get(performance_sample_key(visual_row), [])
            match_strengths = {
                primary_index: performance_condition_match_strength(primary.loc[primary_index], visual_row)
                for primary_index in sample_primary
            }
            compatible_primary = [
                primary_index for primary_index, strength in match_strengths.items() if strength > 0
            ]
            if not compatible_primary and len(sample_primary) == 1:
                only_primary = sample_primary[0]
                if match_strengths[only_primary] == 0:
                    compatible_primary = [only_primary]
            if not compatible_primary:
                continue
            visual_candidates[visual_index] = compatible_primary
            for primary_index in compatible_primary:
                primary_candidate_counts[primary_index] += 1

        new_pairs = []
        for visual_index, compatible_primary in visual_candidates.items():
            if len(compatible_primary) != 1:
                continue
            primary_index = compatible_primary[0]
            if primary_candidate_counts[primary_index] == 1:
                new_pairs.append((primary_index, visual_index))
        if not new_pairs:
            break
        for primary_index, visual_index in new_pairs:
            if primary_index in pairings or visual_index in used_visual_indices:
                continue
            pairings[primary_index] = (visual_index, visual.loc[visual_index])
            used_visual_indices.add(visual_index)

    return pairings


def ensure_unique_performance_record_uids(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    uid_column = "performance_record_uid"
    if frame.empty or uid_column not in frame.columns:
        return frame, 0

    result = frame.copy()
    duplicate_mask = result[uid_column].map(clean_value).duplicated(keep=False)
    if not duplicate_mask.any():
        return result, 0

    occupied = set(result.loc[~duplicate_mask, uid_column].dropna().astype(str))
    changed = 0
    for base_uid, group in result.loc[duplicate_mask].groupby(uid_column, sort=False, dropna=False):
        base_text = str(clean_value(base_uid) or "performance-record")
        candidate_counts = defaultdict(int)
        for row_index, row in group.iterrows():
            payload = [
                (column, normalize_exact_dedupe_value(row.get(column)))
                for column in result.columns
                if column != uid_column
            ]
            digest = hashlib.sha1(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:12]
            candidate_base = f"{base_text}::D{digest}"
            candidate_counts[candidate_base] += 1
            occurrence = candidate_counts[candidate_base]
            candidate = candidate_base if occurrence == 1 else f"{candidate_base}::N{occurrence}"
            while candidate in occupied:
                occurrence += 1
                candidate_counts[candidate_base] = occurrence
                candidate = f"{candidate_base}::N{occurrence}"
            result.at[row_index, uid_column] = candidate
            occupied.add(candidate)
            changed += 1

    if result[uid_column].map(clean_value).duplicated().any():
        raise ValueError("performance record UID disambiguation failed")
    return result, changed


def merge_text_and_visual_frames(primary: pd.DataFrame, visual: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    pairings = pair_visual_performance_rows(primary, visual)
    used_visual_indices = {visual_index for visual_index, _ in pairings.values()}

    merged_rows = []
    filled_count = 0

    for primary_index, row in primary.iterrows():
        merged = row.copy()
        paired_visual = pairings.get(primary_index)
        visual_row = paired_visual[1] if paired_visual is not None else None
        supplemented = []
        if visual_row is not None:
            for col in METRIC_COLUMNS:
                if col not in primary.columns or col not in visual.columns:
                    continue
                if value_missing(merged.get(col)) and has_supplement_value(visual_row.get(col)):
                    visual_value = normalize_metric_output_value(visual_row.get(col))
                    merged[col] = visual_value
                    supplemented.append(f"{col}={visual_value}")
                    filled_count += 1
        if supplemented:
            note = clean_value(merged.get(COL_NOTES)) or ""
            visual_note = clean_value(visual_row.get(COL_NOTES)) or ""
            supplement_note = "supplemental performance used: " + "; ".join(supplemented)
            if visual_note:
                supplement_note += f"; supplement notes: {visual_note}"
            merged[COL_NOTES] = f"{note}; {supplement_note}" if note else supplement_note
        merged_rows.append(merged)

    output_columns = list(primary.columns)
    for visual_index, row in visual.iterrows():
        if visual_index in used_visual_indices:
            continue
        if not any(has_supplement_value(row.get(col)) for col in METRIC_COLUMNS if col in visual.columns):
            continue
        new_row = {col: None for col in output_columns}
        visual_only_supplements = []
        for col in output_columns:
            if col in visual.columns:
                value = clean_value(row.get(col))
                if col in METRIC_COLUMNS and has_supplement_value(value):
                    value = normalize_metric_output_value(value)
                    visual_only_supplements.append(f"{col}={value}")
                new_row[col] = value
        note = clean_value(new_row.get(COL_NOTES)) or ""
        supplement_note = "supplement-only row not matched to primary performance table"
        if visual_only_supplements:
            supplement_note += "; supplement values: " + "; ".join(visual_only_supplements)
        new_row[COL_NOTES] = f"{supplement_note}; {note}" if note else supplement_note
        merged_rows.append(pd.Series(new_row))

    merged = pd.DataFrame(merged_rows, columns=output_columns)
    merged, disambiguated_count = ensure_unique_performance_record_uids(merged)
    if disambiguated_count:
        print(f"[identity] disambiguated {disambiguated_count} colliding performance record UIDs")
    return merged, filled_count

def merge_text_and_visual(primary_path: Path, visual_path: Path, output_path: Path) -> None:
    if not primary_path.exists():
        raise FileNotFoundError(f"Missing primary performance file: {primary_path}")
    if not visual_path.exists():
        raise FileNotFoundError(f"Missing visual performance file: {visual_path}")

    primary = read_excel_canonical(primary_path)
    visual = read_excel_canonical(visual_path)
    merged, filled_count = merge_text_and_visual_frames(primary, visual)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(output_path, index=False)
    print(f"Merged performance written: {output_path}")
    print(f"Visual supplements inserted: {filled_count}")


def run_visual(args):
    output_dir = Path(args.output_dir).resolve()
    analyzer = VisualPerformanceAnalyzer(
        output_excel=str(output_dir / "performance_samples_visual.xlsx"),
        model_name=args.model,
        input_folder=str(Path(args.input).resolve()),
    )
    configure_runtime(analyzer, args, make_api_key_queue(load_api_keys(args.api_keys_file)))
    analyzer.process_folder(args.input)
    analyzer.summarize_run()


def run_wide_merge(
    args,
    performance_path: Optional[Path] = None,
    release_safe: bool = False,
) -> None:
    output_dir = Path(args.output_dir).resolve()
    entities_path = output_dir / "formulation_entities.xlsx"
    parts_path = output_dir / "formulation_entity_parts.xlsx"
    performance = performance_path or output_dir / "performance_samples_final.xlsx"
    wide_path = Path(args.merged_wide).resolve() if args.merged_wide else output_dir / "flame_retardant_data_optimized_final.xlsx"
    merge_entity_outputs(
        entities_path,
        parts_path,
        performance,
        wide_path,
        release_safe=release_safe,
    )


def run_merge(args):
    output_dir = Path(args.output_dir).resolve()
    entities_path = output_dir / "formulation_entities.xlsx"
    primary_path = Path(args.primary_performance).resolve() if args.primary_performance else output_dir / "performance_samples.xlsx"
    visual_path = Path(args.visual_performance).resolve() if args.visual_performance else output_dir / "performance_samples_visual.xlsx"
    output_path = Path(args.merged_performance).resolve() if args.merged_performance else output_dir / "performance_samples_final.xlsx"

    if not entities_path.exists():
        print(f"[merge] missing formulation output; skipping merge: {entities_path}")
        return
    if not primary_path.exists():
        print(f"[merge] missing text performance output; skipping merge: {primary_path}")
        return
    if visual_path.exists():
        merge_text_and_visual(primary_path, visual_path, output_path)
    else:
        print(f"[merge] visual performance output is missing; using text performance only: {visual_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        read_excel_canonical(primary_path).to_excel(output_path, index=False)
    run_wide_merge(args, output_path)


REPLAY_RAW_PREFIXES = {
    "formulation": "formulation_entity_raw_",
    "performance": "performance_raw_",
    "visual": "visual_performance_raw_",
}


def replay_input_index(input_folder: str) -> Dict[str, Dict]:
    input_path = Path(input_folder).resolve()
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Replay input folder not found: {input_path}")
    subdirs = sorted(
        name for name in os.listdir(input_path)
        if os.path.isdir(os.path.join(input_path, name))
    )
    return {
        name: {"article_idx": article_idx, "paper_dir": str(input_path / name)}
        for article_idx, name in enumerate(subdirs, 1)
    }


def replay_file_id(path: Path, prefix: str, known_file_ids: set[str]) -> str:
    name = path.name
    tail = name[len(prefix):] if name.startswith(prefix) else path.stem
    if tail.lower().endswith(".json"):
        tail = tail[:-5]
    if tail in known_file_ids:
        return tail
    matches = [
        file_id for file_id in known_file_ids
        if tail.startswith(f"{file_id}.") or tail.startswith(f"{file_id}_parse_error_")
    ]
    if matches:
        return max(matches, key=len)
    return tail


def collect_replay_candidates(raw_dir: Path, bad_raw_dir: Path, input_index: Dict[str, Dict]) -> List[Dict]:
    candidates = []
    known_file_ids = set(input_index)
    for source_kind, directory in [("raw", raw_dir), ("bad", bad_raw_dir)]:
        if not directory.exists() or not directory.is_dir():
            continue
        for task, prefix in REPLAY_RAW_PREFIXES.items():
            for path in sorted(directory.glob(f"{prefix}*.json")):
                file_id = replay_file_id(path, prefix, known_file_ids)
                mapping = input_index.get(file_id)
                candidates.append({
                    "task": task,
                    "file_id": file_id,
                    "article_idx": mapping.get("article_idx") if mapping else None,
                    "paper_dir": mapping.get("paper_dir") if mapping else None,
                    "source_kind": source_kind,
                    "raw_path": str(path.resolve()),
                    "mtime": path.stat().st_mtime,
                    "status": "pending",
                    "selected": False,
                    "row_count": 0,
                    "part_row_count": 0,
                    "error": None,
                    "_rows": [],
                    "_parts": [],
                })
    return candidates


def make_replay_entity_analyzer() -> EntityFormulationAnalyzer:
    analyzer = object.__new__(EntityFormulationAnalyzer)
    analyzer._part_context = threading.local()
    return analyzer


def make_replay_performance_analyzer(
    formulation_cache: pd.DataFrame,
    visual: bool = False,
) -> PerformanceAnalyzer:
    analyzer_class = VisualPerformanceAnalyzer if visual else PerformanceAnalyzer
    analyzer = object.__new__(analyzer_class)
    analyzer._formulation_cache = formulation_cache
    analyzer._formulation_cache_lock = Lock()
    analyzer._formulation_samples_cache = {}
    analyzer._thread_context = threading.local()
    analyzer.formulation_excel = Path("__offline_replay_formulation__.xlsx")
    analyzer.task_name = "visual_performance" if visual else "performance"
    analyzer.log = lambda message: None
    return analyzer


def parse_replay_candidate(candidate: Dict, analyzer) -> None:
    if candidate["article_idx"] is None or not candidate["paper_dir"]:
        candidate["status"] = "unmapped"
        candidate["error"] = "raw file_id is not present in the complete sorted --input directory list"
        return
    try:
        raw_text = Path(candidate["raw_path"]).read_text(encoding="utf-8", errors="replace")
        analyzer._smart_fix_json(raw_text)
        if isinstance(analyzer, EntityFormulationAnalyzer):
            rows = analyzer.parse_task_response(raw_text, candidate["article_idx"])
            rows = analyzer.attach_source_metadata(rows, candidate["file_id"], candidate["paper_dir"])
            parts = [dict(row) for row in analyzer._get_pending_part_rows()]
            analyzer._set_pending_part_rows([])
        else:
            analyzer._thread_context.file_id = candidate["file_id"]
            try:
                rows = analyzer.parse_task_response(raw_text, candidate["article_idx"])
                rows = analyzer.attach_source_metadata(rows, candidate["file_id"], candidate["paper_dir"])
            finally:
                analyzer._thread_context.file_id = None
            parts = []
        candidate["_rows"] = [dict(row) for row in rows]
        candidate["_parts"] = parts
        candidate["row_count"] = len(rows)
        candidate["part_row_count"] = len(parts)
        candidate["status"] = "success" if rows or parts else "empty"
    except Exception as exc:
        candidate["status"] = "rejected"
        candidate["error"] = f"{type(exc).__name__}: {exc}"


def select_replay_candidates(candidates: List[Dict], task: str) -> List[Dict]:
    grouped = defaultdict(list)
    for candidate in candidates:
        if candidate["task"] == task:
            grouped[candidate["file_id"]].append(candidate)

    selected = []
    for file_candidates in grouped.values():
        valid = [candidate for candidate in file_candidates if candidate["status"] in {"success", "empty"}]
        if not valid:
            continue
        chosen = sorted(
            valid,
            key=lambda candidate: (
                0 if candidate["status"] == "success" else 1,
                0 if candidate["source_kind"] == "raw" else 1,
                -candidate["mtime"],
            ),
        )[0]
        chosen["selected"] = True
        selected.append(chosen)
    return selected


def replay_dataframe(rows: List[Dict], columns: List[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).reindex(columns=columns)


def run_replay(args) -> None:
    output_dir = Path(args.output_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve() if args.raw_dir else output_dir / "api_raw"
    bad_raw_dir = Path(args.bad_raw_dir).resolve() if args.bad_raw_dir else output_dir / "api_raw_bad"
    replay_output_dir = (
        Path(args.replay_output_dir).resolve()
        if args.replay_output_dir
        else output_dir / f"replay_csv_{datetime.now():%Y%m%d_%H%M%S}"
    )
    replay_report = (
        Path(args.replay_report).resolve()
        if args.replay_report
        else replay_output_dir / "replay_report.csv"
    )
    if replay_report.suffix.lower() != ".csv":
        raise ValueError("--replay-report must be a CSV path")

    input_index = replay_input_index(args.input)
    candidates = collect_replay_candidates(raw_dir, bad_raw_dir, input_index)

    entity_analyzer = make_replay_entity_analyzer()
    for candidate in candidates:
        if candidate["task"] == "formulation":
            parse_replay_candidate(candidate, entity_analyzer)
    selected_formulation = select_replay_candidates(candidates, "formulation")
    entity_rows = [row for candidate in selected_formulation for row in candidate["_rows"]]
    part_rows = [row for candidate in selected_formulation for row in candidate["_parts"]]
    entities = replay_dataframe(entity_rows, ENTITY_COLUMNS)
    parts = replay_dataframe(part_rows, PART_COLUMNS)

    text_analyzer = make_replay_performance_analyzer(entities, visual=False)
    visual_analyzer = make_replay_performance_analyzer(entities, visual=True)
    for candidate in candidates:
        if candidate["task"] == "performance":
            parse_replay_candidate(candidate, text_analyzer)
        elif candidate["task"] == "visual":
            parse_replay_candidate(candidate, visual_analyzer)

    selected_text = select_replay_candidates(candidates, "performance")
    selected_visual = select_replay_candidates(candidates, "visual")
    text_rows = [row for candidate in selected_text for row in candidate["_rows"]]
    visual_rows = [row for candidate in selected_visual for row in candidate["_rows"]]
    performance_text = replay_dataframe(text_rows, PERFORMANCE_COLUMNS)
    performance_visual = replay_dataframe(visual_rows, PERFORMANCE_COLUMNS)
    performance_final, filled_count = merge_text_and_visual_frames(performance_text, performance_visual)

    replay_output_dir.mkdir(parents=True, exist_ok=True)
    replay_report.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "formulation_entities.csv": entities,
        "formulation_entity_parts.csv": parts,
        "performance_samples_text.csv": performance_text,
        "performance_samples_visual.csv": performance_visual,
        "performance_samples_final.csv": performance_final,
    }
    for filename, dataframe in outputs.items():
        dataframe.to_csv(replay_output_dir / filename, index=False, encoding="utf-8-sig")

    report_columns = [
        "task", "file_id", "article_idx", "source_kind", "raw_path", "status",
        "selected", "row_count", "part_row_count", "error",
    ]
    report_rows = [{column: candidate.get(column) for column in report_columns} for candidate in candidates]
    pd.DataFrame(report_rows, columns=report_columns).to_csv(replay_report, index=False, encoding="utf-8-sig")
    status_counts = pd.Series([candidate["status"] for candidate in candidates]).value_counts().to_dict()
    print(f"Offline replay completed: {replay_output_dir}")
    print(f"Replay report: {replay_report}")
    print(f"Candidates: {len(candidates)}; status={status_counts}")
    print(
        f"Rows: entities={len(entities)}, parts={len(parts)}, text={len(performance_text)}, "
        f"visual={len(performance_visual)}, final={len(performance_final)}; visual fills={filled_count}"
    )


def clear_pipeline_outputs(output_dir: Path):
    output_files = [
        output_dir / "formulation_entities.xlsx",
        output_dir / "formulation_entity_parts.xlsx",
        output_dir / "performance_samples.xlsx",
        output_dir / "performance_samples_visual.xlsx",
        output_dir / "performance_samples_final.xlsx",
        output_dir / "flame_retardant_data_optimized.xlsx",
        output_dir / "flame_retardant_data_optimized_final.xlsx",
        output_dir / "run.log",
    ]
    for path in output_files:
        if path.exists() and path.is_file():
            path.unlink()

    raw_dir = output_dir / "api_raw"
    if raw_dir.exists():
        for raw_file in raw_dir.glob("*.json"):
            if raw_file.is_file():
                raw_file.unlink()


def parse_args():
    parser = argparse.ArgumentParser(description="Split EVA/PE/POE extraction into entity formulation and performance tasks.")
    parser.add_argument(
        "--task",
        choices=["formulation", "performance", "performance_v", "merge", "wide", "replay", "all"],
        default="formulation",
        help="Task to run: formulation, text performance, visual performance, final merge, release-safe wide reconstruction, offline raw replay, or all tasks in order. Default: formulation.",
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT_DIR), help=f"MinerU paper folder. Default: {DEFAULT_INPUT_DIR}")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help=f"Output folder. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Chat model name. Default: {DEFAULT_MODEL}")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=f"OpenAI-compatible chat completions endpoint. Default: {DEFAULT_API_URL}")
    parser.add_argument("--force-vision", action="store_true", help="Force sending vision-format image inputs even if model name does not look multimodal.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N paper folders. Default: 0 means all.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent paper requests per task. Default: 1.")
    parser.add_argument(
        "--api-keys-file",
        default=str(DEFAULT_API_KEYS_FILE) if DEFAULT_API_KEYS_FILE.exists() else None,
        help="Text file with one API key per line. Defaults to OPENAI_API_KEYS_FILE or .secrets/api_keys.txt when present.",
    )
    parser.add_argument("--api-retries", type=int, default=4, help="API attempts per paper. Default: 4.")
    parser.add_argument("--retry-base-delay", type=float, default=20, help="Initial retry delay in seconds, doubled each attempt. Default: 20.")
    parser.add_argument("--rate-limit-sleep", type=float, default=1800, help="Seconds to wait after a rate-limit response. Default: 1800.")
    parser.add_argument("--rate-limit-retries", type=int, default=3, help="Rate-limit wait/retry cycles per paper/key. Default: 3.")
    parser.add_argument("--read-timeout", type=float, default=540, help="API read timeout in seconds. Default: 540.")
    parser.add_argument("--max-tokens", type=int, default=65536, help="Maximum output tokens per API request. Default: 65536.")
    parser.add_argument("--stream", action="store_true", help="Request streaming chat output and assemble streamed chunks. Helps avoid gateway timeout during long generation.")
    parser.add_argument("--dry-run", action="store_true", help="Generate API previews without calling the API.")
    parser.add_argument("--force", action="store_true", help="Force rerun current task by deleting that task's existing raw and rows for each processed paper.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing pipeline outputs in output-dir before running.")
    parser.add_argument("--primary-performance", default=None, help="Text performance table path. Default: output-dir/performance_samples.xlsx")
    parser.add_argument("--visual-performance", default=None, help="Visual performance table path. Default: output-dir/performance_samples_visual.xlsx")
    parser.add_argument("--merged-performance", default=None, help="Merged performance table path. Default: output-dir/performance_samples_final.xlsx")
    parser.add_argument("--merged-wide", default=None, help="Final wide-table path. Default: output-dir/flame_retardant_data_optimized_final.xlsx")
    parser.add_argument("--raw-dir", default=None, help="Offline replay source api_raw directory. Default: output-dir/api_raw")
    parser.add_argument("--bad-raw-dir", default=None, help="Offline replay source api_raw_bad directory. Default: output-dir/api_raw_bad")
    parser.add_argument("--replay-output-dir", default=None, help="Offline replay CSV output directory. Default: timestamped directory under output-dir")
    parser.add_argument("--replay-report", default=None, help="Offline replay CSV report path. Default: replay-output-dir/replay_report.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and not args.dry_run and args.task not in {"replay", "wide"}:
        clear_pipeline_outputs(output_dir)

    if args.task == "formulation":
        run_formulation(args)
    elif args.task == "performance":
        run_performance(args)
    elif args.task == "performance_v":
        run_visual(args)
    elif args.task == "merge" and not args.dry_run:
        run_merge(args)
    elif args.task == "wide" and not args.dry_run:
        run_wide_merge(args, release_safe=True)
    elif args.task == "replay":
        run_replay(args)
    elif args.task == "all":
        run_formulation(args)
        run_performance(args)
        run_visual(args)
        if not args.dry_run:
            run_merge(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)

import os
import re
import ast
import time
import base64
import argparse
import mimetypes
import requests
import pandas as pd
from pathlib import Path

try:
    from public_utils import console_print as print, install_ascii_exception_hook
    from schema import canonicalize_dataframe_columns
except ModuleNotFoundError:
    from pipeline.public_utils import console_print as print, install_ascii_exception_hook
    from pipeline.schema import canonicalize_dataframe_columns

install_ascii_exception_hook()

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

TARGET_FIELD_SPECS = [
    ("flame_retardant_1", "flame_retardant_1_figure_locator", "flame_retardant_1_bbox", "flame_retardant_1_smiles", "flame_retardant_1_smiles_source"),
    ("flame_retardant_2", "flame_retardant_2_figure_locator", "flame_retardant_2_bbox", "flame_retardant_2_smiles", "flame_retardant_2_smiles_source"),
    ("flame_retardant_1_modifier", "flame_retardant_1_modifier_figure_locator", "flame_retardant_1_modifier_bbox", "flame_retardant_1_modifier_smiles", "flame_retardant_1_modifier_smiles_source"),
    ("flame_retardant_2_modifier", "flame_retardant_2_modifier_figure_locator", "flame_retardant_2_modifier_bbox", "flame_retardant_2_modifier_smiles", "flame_retardant_2_modifier_smiles_source"),
    ("flame_retardant_1_modified_product_name", "flame_retardant_1_figure_locator", "flame_retardant_1_bbox", "flame_retardant_1_modified_product_smiles", "flame_retardant_1_modified_product_smiles_source"),
    ("flame_retardant_2_modified_product_name", "flame_retardant_2_figure_locator", "flame_retardant_2_bbox", "flame_retardant_2_modified_product_smiles", "flame_retardant_2_modified_product_smiles_source"),
    ("entity_name", "entity_figure_locator", "entity_bbox", "entity_smiles", "entity_smiles_source"),
    ("part_name", "part_figure_locator", "part_bbox", "part_smiles", "part_smiles_source"),
    ("component_name", "component_figure_locator", "component_bbox", "component_smiles", "component_smiles_source"),
    ("composite_or_modified_product_name", "component_figure_locator", "component_bbox", "composite_or_modified_product_smiles", "composite_or_modified_product_smiles_source"),
]

WIDE_FIELD_NAMES = {
    "flame_retardant_1", "flame_retardant_2", "flame_retardant_1_modifier", "flame_retardant_2_modifier",
    "flame_retardant_1_modified_product_name", "flame_retardant_2_modified_product_name",
}


NON_MOLECULAR_NAME_RE = re.compile(
    r"\b(?:ath|mdh|mh|app|mpp|mca|eg|cnts?|mwcnts?|swcnts?|go|rgo|graphene|"
    r"graphite|zinc borate|zb|ldh|mmt|montmorillonite|clay|talc|sio2|silica|"
    r"tio2|al2o3|mgo|zno|fe2o3|red phosphorus|rp|mxene|zif|uio|mof|cof)\b",
    re.IGNORECASE,
)

IMAGE_ID_RE = re.compile(
    r"ID:\s*([^\s,;|)]+\.(?:png|jpg|jpeg|webp|gif|bmp|tiff))",
    re.IGNORECASE,
)

VISUAL_POSITIVE_RE = re.compile(
    r"(graft|grafted|grafting|synthesi[sz]ed|synthesis|synthetic|reaction|scheme|"
    r"new\s+(?:flame\s+)?retardant|novel|designed|derived|derivative|intermediate|"
    r"product\s+structure|chemical\s+structure|molecular\s+structure|repeat(?:ing)?\s+unit|"
    r"dopo|phosphazene|triazine|triazole|hyperbranched|oligomer|polymeric\s+flame|"
    r"organophosph|p-n|p\s*/\s*n|p\s*-?\s*n|p\s*-?\s*n\s*-?\s*s)",
    re.IGNORECASE,
)

HARD_VISUAL_SKIP_RE = re.compile(
    r"(core[-\s]?shell|shell[-\s]?core|microencapsulat|encapsulat|coated|coating|"
    r"surface[-\s]?treated|surface\s+modified|intercalat|doped|in[-\s]?situ\s+grown|"
    r"morphology|sem|tem|xrd|tga|ftir|dtg|xps|dsc|eds|mapping)",
    re.IGNORECASE,
)

SOFT_VISUAL_SKIP_RE = re.compile(
    r"(hybrid|composite|complex|particle|nanoparticle|filler)",
    re.IGNORECASE,
)

STRUCTURE_LOCATOR_RE = re.compile(
    r"(structure|chemical|molecular|reaction|scheme|synthesis|synthetic|formula)",
    re.IGNORECASE,
)

NON_STRUCTURE_LOCATOR_RE = re.compile(
    r"(sem|tem|morphology|xrd|tga|dtg|ftir|spectrum|spectra|xps|dsc|eds|mapping)",
    re.IGNORECASE,
)

ORGANIC_STRUCTURE_MARKER_RE = re.compile(
    r"(dopo|phosphazene|triazine|triazole|silane|kh-\d+|ligand|monomer|"
    r"phosphate|phosphonate|phosphinate|piperazine|melamine|chitosan|lignin|"
    r"tannic|hyperbranched|oligomer|polymer|organic)",
    re.IGNORECASE,
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")
RETRYABLE_VISUAL_STATUS_RE = re.compile(r"api\s*error|network|timeout|rate.limit|http\s*5\d\d", re.IGNORECASE)


def is_empty_cell(value):
    return pd.isna(value) or str(value).strip() in {"", "#", "nan", "None"}


def ensure_column(df, column_name):
    if column_name not in df.columns:
        df[column_name] = pd.NA
    else:
        df[column_name] = df[column_name].astype("object")


def available_target_specs(df, allow_wide=False):
    specs = []
    for spec in TARGET_FIELD_SPECS:
        name_col, locator_col = spec[0], spec[1]
        if name_col not in df.columns or locator_col not in df.columns:
            continue
        if not allow_wide and name_col in WIDE_FIELD_NAMES:
            continue
        specs.append(spec)
    return specs


def clean_model_text(text):
    cleaned = str(text or "").strip()
    cleaned = cleaned.replace("```json", "").replace("```smiles", "").replace("```", "")
    cleaned = cleaned.replace("SMILES:", "").replace("SMILES", "").strip()
    return cleaned.strip("'\"").strip()


def looks_like_smiles(text):
    if is_empty_cell(text):
        return False
    value = str(text).strip()
    normalized_status = re.sub(r"[^A-Za-z]+", "_", value).strip("_").upper()
    if normalized_status in {"NA", "N_A", "NONE", "FAILED", "NO_SMILES", "NOT_MOLECULAR", "INSUFFICIENT_STRUCTURE"}:
        return False
    if any(token in normalized_status for token in ["NO_SMILES", "NOT_MOLECULAR", "INSUFFICIENT_STRUCTURE", "CANNOT_DETERMINE", "UNCLEAR"]):
        return False
    if len(value) > 500 or any(ch.isspace() for ch in value):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9\[\]\(\)=#@+\-\\/%.]+", value))


def joined_row_context(row):
    context_columns = [
        "entity_name", "entity_role", "entity_type", "entity_material_type", "entity_profile", "entity_figure_locator",
        "part_name", "part_role", "part_relation", "modification_method", "part_figure_locator", "part_notes",
        "component_name", "component_figure_locator", "composite_or_modified_product_name",
    ]
    return " | ".join(str(row.get(col, "")).strip() for col in context_columns if col in row.index)


def is_retryable_visual_status(status):
    return bool(RETRYABLE_VISUAL_STATUS_RE.search(str(status or "")))


def detect_image_mime(image_path):
    with open(image_path, "rb") as image_file:
        header = image_file.read(16)
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if header.startswith(b"BM"):
        return "image/bmp"
    if header[:4] in {b"II*\x00", b"MM\x00*"}:
        return "image/tiff"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    guessed, _ = mimetypes.guess_type(str(image_path))
    if guessed and guessed.startswith("image/"):
        return guessed
    raise ValueError(f"Unsupported or unrecognized image type: {image_path}")


def should_try_visual(row, name_col, source_col, locator_col, force=False, retry_errors=False):
    target_name = str(row.get(name_col, "")).strip()
    if not target_name or target_name in {"#", "nan", "None"}:
        return False

    status = str(row.get(source_col, "")).strip()
    locator = str(row.get(locator_col, "")).strip()
    if not force:
        if "Expert Dictionary" in status or "Sniped" in status or "Success" in status:
            return False
        if status != "Pending Visual Extraction" and not (retry_errors and is_retryable_visual_status(status)):
            return False
    if not re.search(r"\bID\s*:", locator, re.IGNORECASE):
        return False

    context = f"{target_name} | {locator} | {joined_row_context(row)}"
    positive = bool(VISUAL_POSITIVE_RE.search(context))
    structure_locator = bool(STRUCTURE_LOCATOR_RE.search(locator))
    non_structure_locator = bool(NON_STRUCTURE_LOCATOR_RE.search(locator))
    organic_marker = bool(ORGANIC_STRUCTURE_MARKER_RE.search(context))
    common_non_molecular = bool(NON_MOLECULAR_NAME_RE.search(target_name))
    hard_skip = bool(HARD_VISUAL_SKIP_RE.search(context))
    soft_skip = bool(SOFT_VISUAL_SKIP_RE.search(context))

    if hard_skip and not re.search(r"(graft|grafted)", context, re.IGNORECASE):
        return False
    if soft_skip and not (positive and organic_marker):
        return False
    if common_non_molecular and not organic_marker:
        return False
    if non_structure_locator and not structure_locator:
        return False
    if not (positive or structure_locator):
        return False
    if not organic_marker and not re.search(r"(graft|grafted|synthesi[sz]ed)", context, re.IGNORECASE):
        return False

    return True

class MLLMContextSniper:
    def __init__(self, api_base=None, api_keys=None, model_name=None, api_format="auto", request_timeout=300):
        print("Initializing the lightweight RAG vision extractor...")
        
        self.api_base = api_base or os.getenv(
            "MLLM_API_BASE",
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"),
        )
        env_key = os.getenv("MLLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.api_keys = [key.strip() for key in (api_keys or split_api_keys(env_key)) if key and key.strip()]
        self.model_name = model_name or os.getenv("MLLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5.5")
        self.request_timeout = request_timeout
        self._api_key_index = 0
        if api_format == "auto":
            self.api_format = "openai" if "/chat/completions" in self.api_base.lower() else "gemini"
        else:
            self.api_format = api_format
        if not self.api_keys:
            raise ValueError("Missing vision-model API key. Set MLLM_API_KEY, OPENAI_API_KEY, or use --api-key-file.")
        print(f"Vision API initialized | format: {self.api_format} | model: {self.model_name}")


    def get_surrounding_context(self, image_path, image_filename, context_window=1500):
        """
        Find adjacent MinerU Markdown and return context around an image reference.
        """
        if not image_path:
            return "No text context found."
            
        #  .../Paper_id/images/img.png
        #  md  image_path
        image_dir = os.path.dirname(image_path)
        parent_dir = os.path.dirname(image_dir)
        
        md_path = None
        #  .md
        for search_dir in [image_dir, parent_dir]:
            if not os.path.exists(search_dir): continue
            for file in os.listdir(search_dir):
                if file.endswith(".md"):
                    md_path = os.path.join(search_dir, file)
                    break
            if md_path: break
            
        if not md_path:
            return "No text context found in adjacent folders."

        #
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if image_filename in content:
                idx = content.find(image_filename)
                start_idx = max(0, idx - context_window)
                end_idx = min(len(content), idx + context_window)
                context = content[start_idx:end_idx]
                
                #
                context = re.sub(r'^[^\w]+', '', context).strip()
                return context
            else:
                return "Image filename not mentioned in the adjacent markdown."
        except Exception as e:
            return f"Error reading markdown: {e}"

    def encode_image(self, image_path):
        """Encode an image as Base64 text."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def build_image_index(self, image_folder):
        image_index = {}
        for root, dirs, files in os.walk(image_folder):
            for filename in files:
                lower = filename.lower()
                if lower.endswith(IMAGE_EXTENSIONS) and lower not in image_index:
                    image_index[lower] = os.path.join(root, filename)
        return image_index

    def next_api_key(self):
        key = self.api_keys[self._api_key_index % len(self.api_keys)]
        self._api_key_index += 1
        return key

    def build_payload(self, prompt, base64_image, mime_type):
        if self.api_format == "openai":
            return {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.0,
            }

        return {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"data": base64_image, "mimeType": mime_type}}
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 150,
                "temperature": 0.0
            }
        }

    def extract_answer_text(self, response_json):
        if self.api_format == "openai":
            message = response_json.get("choices", [{}])[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        parts.append(str(item.get("text", "")))
                    else:
                        parts.append(str(item))
                return "\n".join(part for part in parts if part)
            return str(content or "")

        return response_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

    def snipe_smiles(self, target_name, img_path, bbox_hint, context_text):
        """Identify one visual chemical structure and return a SMILES or failure reason."""
        base64_image = self.encode_image(img_path)
        mime_type = detect_image_mime(img_path)
        
        prompt = (
            f"TASK: Identify the chemical structure for target '{target_name}' from the figure and paper context.\n"
            f"Return ONLY one line.\n\n"
            f"Allowed outputs:\n"
            f"1. A valid SMILES string, only if the final target molecule, repeating unit, ligand, monomer, or grafted product structure is visibly shown or unambiguously specified.\n"
            f"2. NO_SMILES if the target is an inorganic particle, mineral filler, polymer particle, coating/core-shell composite, morphology image, SEM/TEM/XRD/TGA curve, or a material without a determinate molecular graph.\n"
            f"3. INSUFFICIENT_STRUCTURE if the figure/context shows only precursors, a reaction arrow without final structure, labels without structure, or the target region is unclear.\n\n"
            f"Strict rules:\n"
            f"- Do NOT invent a grafted-product SMILES by combining precursor/monomer SMILES.\n"
            f"- For grafted/newly synthesized additives, output the final product/repeating-unit SMILES only when that final structure is shown.\n"
            f"- For surface-treated, coated, core-shell, encapsulated, intercalated, or doped particles, output NO_SMILES unless a discrete molecular component structure is explicitly shown for this exact target.\n"
            f"- If the target name refers to a part such as shell, ligand, grafting monomer, coupling agent, or modifier, output that part's SMILES only when its own molecular structure is visible or explicit.\n"
            f"- Do not output explanations, markdown, labels, or confidence text.\n\n"
            f"--- SUPPLEMENTARY INFORMATION ---\n"
            f"Bounding Box Hint (Relative coordinates): {bbox_hint if bbox_hint else 'None provided. Scan the whole image.'}\n"
            f"Context from the paper text:\n\"\"\"{context_text}\"\"\""
        )
        
        payload = self.build_payload(prompt, base64_image, mime_type)

        attempts = max(1, len(self.api_keys))
        for attempt in range(attempts):
            api_key = self.next_api_key()
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            try:
                response = requests.post(self.api_base, headers=headers, json=payload, timeout=self.request_timeout)
                if response.status_code == 200:
                    res_data = response.json()
                    answer = self.extract_answer_text(res_data)
                    smiles = clean_model_text(answer)

                    normalized_answer = re.sub(r"[^A-Za-z]+", "_", smiles).strip("_").upper()
                    if normalized_answer in {"NO_SMILES", "NOT_MOLECULAR"}:
                        print(f"[Non-molecular structure] {target_name}")
                        return "Failed", "MLLM No Determinate Molecular SMILES"
                    if normalized_answer == "INSUFFICIENT_STRUCTURE":
                        print(f"[Insufficient structure evidence] {target_name}")
                        return "Failed", "MLLM Insufficient Structure"
                    if looks_like_smiles(smiles):
                        print(f"[Vision extraction succeeded] {target_name} -> {smiles}")
                        return smiles, "MLLM RAG Sniped"
                    elif smiles:
                        print(f"[Output is not a reliable SMILES] {target_name} -> {smiles[:80]}")
                        return "Failed", "MLLM Invalid SMILES Output"
                    else:
                        print("[Vision extraction error] The model returned empty content")
                        return "Failed", "MLLM Output Empty"
                else:
                    print(f"API error (attempt {attempt + 1}/{attempts}): {response.status_code} {response.text[:100]}")
                    time.sleep(3)
            except Exception as e:
                print(f"Network error (attempt {attempt + 1}/{attempts}): {e}")
                time.sleep(5)
                
        return "Failed", "API Error"

    def process_excel(self, excel_input, image_folder, excel_output, allow_wide=False, force=False, retry_errors=False):
        if os.path.exists(excel_output):
            print("\nExisting progress detected; resuming...")
            df = canonicalize_dataframe_columns(pd.read_excel(excel_output), context=str(excel_output))
        else:
            print("\nStarting a new vision extraction run...")
            df = canonicalize_dataframe_columns(pd.read_excel(excel_input), context=str(excel_input))

        total_processed = 0
        total_success = 0
        
        # Reuse successful results for repeated material-image-bbox requests.
        memory_cache = {}

        target_specs = available_target_specs(df, allow_wide=allow_wide)
        print(f"Vision target fields: {', '.join(spec[0] for spec in target_specs)}")
        if not target_specs:
            print("No eligible entity/part vision fields. Wide tables are skipped unless --allow-wide is set.")
            os.makedirs(os.path.dirname(excel_output), exist_ok=True)
            df.to_excel(excel_output, index=False)
            return

        print("Building the image index...")
        image_index = self.build_image_index(image_folder)
        print(f"Indexed images: {len(image_index)}")

        for index, row in df.iterrows():
            for name_col, locator_col, bbox_col, smiles_col, source_col in target_specs:
                ensure_column(df, smiles_col)
                ensure_column(df, source_col)
                status = str(row.get(source_col, ""))
                target_name = str(row.get(name_col, "")).strip()
                
                # Keep completed rows unless an explicit rerun was requested.
                if not force and ("Expert Dictionary" in status or "Sniped" in status or "Success" in status):
                    continue

                if should_try_visual(
                    row,
                    name_col,
                    source_col,
                    locator_col,
                    force=force,
                    retry_errors=retry_errors,
                ):
                    
                    img_locator = str(row.get(locator_col, ""))
                    match = IMAGE_ID_RE.search(img_locator)
                    if not match:
                        continue
                    image_filename = match.group(1)

                    # Parse an optional extraction bounding box.
                    bbox_str = str(row.get(bbox_col, "")).strip() if bbox_col in df.columns else ""
                    pre_bbox = None
                    if bbox_str and bbox_str.lower() != "nan":
                        try:
                            pre_bbox = ast.literal_eval(bbox_str)
                        except (SyntaxError, ValueError):
                            pre_bbox = None

                    cache_key = f"{name_col}|{target_name.lower()}|{image_filename.lower()}|{pre_bbox}"
                    if cache_key in memory_cache:
                        cached_smiles, cached_reason = memory_cache[cache_key]
                        print("-" * 40)
                        print(f"Memory cache hit: row {index + 1} | target: {target_name} | reused")
                        if cached_smiles and cached_smiles != "Failed":
                            df.at[index, smiles_col] = cached_smiles
                            total_success += 1
                        df.at[index, source_col] = cached_reason
                        total_processed += 1
                        os.makedirs(os.path.dirname(excel_output), exist_ok=True)
                        df.to_excel(excel_output, index=False)
                        continue
                    
                    # Resolve the image referenced by the evidence locator.
                    image_path = image_index.get(image_filename.lower())
                    if not image_path:
                        print(f"Image file not found: {image_filename}")
                        continue
                    
                    # Add nearby Markdown text as supporting context.
                    context_text = self.get_surrounding_context(image_path, image_filename, context_window=1500)
                        
                    print("-" * 40)
                    print(f"Vision structure extraction: row {index + 1} | field: {name_col} | target: {target_name} | image: {image_filename}")
                    
                    if force or (retry_errors and is_retryable_visual_status(status)):
                        df.at[index, smiles_col] = pd.NA
                    smiles, reason = self.snipe_smiles(target_name, image_path, pre_bbox, context_text)
                    
                    if not is_retryable_visual_status(reason):
                        memory_cache[cache_key] = (smiles, reason)
                    
                    if smiles != "Failed":
                        df.at[index, smiles_col] = smiles
                        total_success += 1
                    df.at[index, source_col] = reason
                    total_processed += 1
                    
                    # Save after each request so interrupted runs can resume.
                    os.makedirs(os.path.dirname(excel_output), exist_ok=True)
                    df.to_excel(excel_output, index=False)
                    time.sleep(1)

        print("-" * 40)
        print("Vision structure extraction completed.")
        print(f"Vision requests: {total_processed}; reliable SMILES: {total_success}.")
        os.makedirs(os.path.dirname(excel_output), exist_ok=True)
        df.to_excel(excel_output, index=False)


def split_api_keys(raw_value):
    if not raw_value:
        return []
    return [
        item.strip()
        for item in re.split(r"[\r\n,;]+", str(raw_value))
        if item.strip()
    ]


def read_api_key_file(path):
    if not path:
        return []
    key_path = Path(path)
    if not key_path.exists():
        raise FileNotFoundError(f"API key file does not exist: {key_path}")
    keys = [
        line.strip()
        for line in key_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if keys:
        return keys
    raise ValueError(f"API key file is empty: {key_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use local MinerU images and an MLLM to fill pending flame-retardant SMILES.")
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "outputs" / "pipeline_normalized" / "formulation_entities.xlsx"),
        help="Input Excel after dictionary cleaning.",
    )
    parser.add_argument(
        "--image-folder",
        default=str(REPO_ROOT / "inputs" / "mineru"),
        help="Parent folder containing MinerU paper outputs.",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "outputs" / "pipeline_normalized" / "formulation_entities_visual.xlsx"),
        help="Output Excel with visually extracted SMILES.",
    )
    parser.add_argument("--allow-wide", action="store_true", help="Allow scanning duplicated final wide-table fields. Default is off to control cost.")
    parser.add_argument("--api-base", default=None, help="Vision API endpoint. Default reads MLLM_API_BASE or OPENAI_BASE_URL.")
    parser.add_argument("--api-key-file", default=None, help="Text file containing one or more vision API keys, one key per line.")
    parser.add_argument("--api-format", choices=["auto", "gemini", "openai"], default="auto", help="Vision API payload format. Auto uses openai for /chat/completions, otherwise gemini.")
    parser.add_argument("--request-timeout", type=float, default=300.0, help="Per visual request timeout in seconds.")
    parser.add_argument("--model", default=None, help="Vision model name label for logging.")
    parser.add_argument("--force", action="store_true", help="Re-run all otherwise eligible visual targets, including prior successes.")
    parser.add_argument("--retry-errors", action="store_true", help="Retry rows whose prior source status is a transient API/network error.")
    args = parser.parse_args()
    
    # Input workbook containing image locators and pending SMILES fields.
    input_excel_path = args.input
    
    # Parent directory containing MinerU paper outputs.
    paper_folder_path = args.image_folder
    
    # Output workbook with visual SMILES enrichment.
    output_excel_path = args.output
    
    # Preserve long-path compatibility on Windows.
    paper_folder_path = os.path.abspath(paper_folder_path)
    if os.name == 'nt' and not paper_folder_path.startswith('\\\\\\'):
        paper_folder_path = '\\\\\\' + paper_folder_path
    
    #
    api_keys = read_api_key_file(args.api_key_file)
    sniper = MLLMContextSniper(
        api_base=args.api_base,
        api_keys=api_keys,
        model_name=args.model,
        api_format=args.api_format,
        request_timeout=args.request_timeout,
    )
    
    sniper.process_excel(
        excel_input=input_excel_path,
        image_folder=paper_folder_path,
        excel_output=output_excel_path,
        allow_wide=args.allow_wide,
        force=args.force,
        retry_errors=args.retry_errors,
    )

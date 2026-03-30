import os
import tarfile
import json
import subprocess
import sys
import numpy as np
import torch
import onnx
import onnxruntime as ort
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForSpeechSeq2Seq,
    AutoModelForAudioClassification,
    AutoModelForImageClassification,
    AutoProcessor,
    AutoTokenizer,
    AutoFeatureExtractor,
)
from pathlib import Path


def download_and_convert_to_onnx(
    model_id: str,
    output_dir: str = "./onnx_export",
    opset_version: int = 21,
    dynamic_axes: dict | None = None,
    dummy_input_override: dict | None = None,
    model_class=None,
    verify: bool = True,
    export_encoder_only: bool | None = None,
    software_spec: dict | None = None,
    hardware_spec: dict | None = None,
):
    """
    Downloads a PyTorch model from Hugging Face and converts it to ONNX format
    ready for upload to watsonx.ai Runtime.

    Parameters
    ----------
    model_id : str
        Hugging Face model identifier (e.g. "openai/whisper-small") or a URL.
        If a full URL is given, the repo id is extracted automatically.
    output_dir : str
        Directory where the .onnx file and .tar.gz archive will be written.
    opset_version : int
        ONNX opset version. watsonx.ai supports opset 19 (onnxruntime 1.16) and
        opset 21 (onnxruntime 1.17). Default is 21 for the latest non-deprecated spec.
    dynamic_axes : dict | None
        Optional dynamic axes mapping passed to torch.onnx.export.
        If None, a sensible default is inferred (batch + sequence dimensions).
    dummy_input_override : dict | None
        Optional dictionary of {name: tensor} to use as the dummy input instead of
        the auto-generated one. Use this for non-standard model architectures.
    model_class : class | None
        Optional transformers Auto* class to load the model (e.g. AutoModelForCTC).
        If None, the best class is auto-detected from the model config.
    verify : bool
        If True, run onnxruntime inference on the exported model to sanity-check it.
    export_encoder_only : bool | None
        For encoder-decoder models (Whisper, T5, BART, etc.):
          - True  = export only the encoder (recommended for watsonx.ai scoring).
          - False = export the full model (requires decoder_input_ids in dummy input).
          - None  = auto-detect: defaults to True for encoder-decoder architectures.
    software_spec : dict | None
        Override for the watsonx.ai software specification. Dict with "name" and/or "id".
        Default: {"name": "onnxruntime_opset_21", "id": "d0b5361f-0810-554e-8d7b-874b5ff799b1"}
        for opset >= 20, or {"name": "onnxruntime_opset_19"} for opset <= 19.
    hardware_spec : dict | None
        Override for the watsonx.ai hardware specification. Dict with "name" and/or "id".
        Default: {"name": "L", "id": "a6c4923b-b8e4-444c-9f43-8a7ec3020110"}

    Returns
    -------
    dict with keys:
        "onnx_path"      – path to the exported .onnx file
        "tar_gz_path"    – path to the .tar.gz archive ready for watsonx.ai upload
        "model_type"     – suggested watsonx.ai model type string
        "sw_spec"        – software specification dict (name + id)
        "hw_spec"        – hardware specification dict (name + id)
        "input_schema"   – watsonx.ai-compatible input schema
        "output_schema"  – watsonx.ai-compatible output schema
    """

    # ------------------------------------------------------------------ #
    # 1. Resolve model_id from URL if needed
    # ------------------------------------------------------------------ #
    if model_id.startswith("http"):
        parts = model_id.rstrip("/").split("huggingface.co/")
        model_id = parts[-1] if len(parts) > 1 else model_id

    print(f"[1/7] Loading config for '{model_id}' ...")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # ------------------------------------------------------------------ #
    # 2. Download model + preprocessor
    # ------------------------------------------------------------------ #
    print(f"[2/7] Downloading model '{model_id}' ...")
    loader = model_class if model_class is not None else _resolve_auto_class(config)
    model = loader.from_pretrained(model_id, trust_remote_code=True)
    model.eval()

    preprocessor = _load_preprocessor(model_id)

    # ------------------------------------------------------------------ #
    # 3. Determine if encoder-decoder and select export target
    # ------------------------------------------------------------------ #
    is_enc_dec = getattr(config, "is_encoder_decoder", False)
    if export_encoder_only is None:
        export_encoder_only = is_enc_dec

    if export_encoder_only and is_enc_dec:
        print("[3/7] Encoder-decoder detected — exporting encoder only ...")
        export_model = _extract_encoder(model)
    else:
        print("[3/7] Exporting full model ...")
        export_model = model

    # ------------------------------------------------------------------ #
    # 4. Build dummy input
    # ------------------------------------------------------------------ #
    print("[4/7] Building dummy input ...")
    if dummy_input_override is not None:
        dummy_input = dummy_input_override
    else:
        dummy_input = _build_dummy_input(
            config, preprocessor, is_enc_dec, export_encoder_only
        )

    input_names = list(dummy_input.keys())
    dummy_tensors = tuple(dummy_input.values())

    # Forward pass to discover output names
    with torch.no_grad():
        sample_out = export_model(**dummy_input)

    output_names = _resolve_output_names(sample_out)

    # ------------------------------------------------------------------ #
    # 5. Dynamic axes
    # ------------------------------------------------------------------ #
    # Names whose non-batch dimensions are fixed (mel bins, image channels, etc.)
    _static_shape_inputs = {"input_features", "pixel_values"}

    if dynamic_axes is None:
        dynamic_axes = {}
        for name in input_names:
            t = dummy_input[name]
            axes = {0: "batch_size"}
            if name not in _static_shape_inputs:
                if t.dim() > 1:
                    axes[1] = "sequence_length"
                if t.dim() > 2:
                    axes[2] = "feature_dim"
            dynamic_axes[name] = axes
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}

    # ------------------------------------------------------------------ #
    # 6. Export to ONNX
    # ------------------------------------------------------------------ #
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    safe_name = model_id.replace("/", "_")
    suffix = "_encoder" if export_encoder_only and is_enc_dec else ""
    onnx_filename = f"{safe_name}{suffix}.onnx"
    onnx_path = out_path / onnx_filename

    print(f"[5/7] Exporting to ONNX (opset {opset_version}) ...")
    torch.onnx.export(
        export_model,
        dummy_tensors,
        str(onnx_path),
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        dynamo=False,
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # Build watsonx.ai-compatible input/output schema from the ONNX graph
    input_schema, output_schema = _build_model_schema(onnx_model)

    del onnx_model
    print(f"       ONNX check passed — saved to {onnx_path}")

    # ------------------------------------------------------------------ #
    # 7. Verify with onnxruntime
    # ------------------------------------------------------------------ #
    if verify:
        print("[6/7] Verifying with onnxruntime ...")
        sess = ort.InferenceSession(str(onnx_path))
        ort_inputs = {k: v.numpy() for k, v in dummy_input.items()}
        ort_outputs = sess.run(None, ort_inputs)
        print(f"       Verification passed — {len(ort_outputs)} output(s) produced.")
        del sess
    else:
        print("[6/7] Skipping verification (verify=False).")

    # ------------------------------------------------------------------ #
    # 8. Package as .tar.gz for watsonx.ai import
    # ------------------------------------------------------------------ #
    print("[7/7] Packaging as .tar.gz ...")
    tar_gz_path = out_path / f"{onnx_filename}.tar.gz"
    with tarfile.open(str(tar_gz_path), "w:gz") as tar:
        tar.add(str(onnx_path), arcname=onnx_filename)

    print(f"       Archive ready at {tar_gz_path}")

    sw_spec, hw_spec, model_type = _resolve_watsonx_spec(
        opset_version, software_spec, hardware_spec
    )

    print("\n=== watsonx.ai upload metadata ===")
    print(f"  Model type            : {model_type}")
    print(f"  Software spec         : {sw_spec['name']} (id: {sw_spec['id']})")
    print(f"  Hardware spec         : {hw_spec['name']} (id: {hw_spec['id']})")
    print(f"  Archive               : {tar_gz_path}")
    print(f"  Input schema fields   : {len(input_schema[0]['fields'])}")
    print(f"  Output schema fields  : {len(output_schema['fields'])}")

    return {
        "onnx_path": str(onnx_path),
        "tar_gz_path": str(tar_gz_path),
        "model_type": model_type,
        "sw_spec": sw_spec,
        "hw_spec": hw_spec,
        "input_schema": input_schema,
        "output_schema": output_schema,
    }


# ====================================================================== #
# Subprocess wrapper (safe for marimo / notebook hot-reload environments)
# ====================================================================== #


def download_and_convert_to_onnx_isolated(
    model_id: str,
    output_dir: str = "./onnx_export",
    opset_version: int = 21,
    export_encoder_only: bool | None = None,
    verify: bool = True,
) -> dict:
    """
    Runs download_and_convert_to_onnx in a subprocess so that torch module
    registration is fully isolated.  Use this from marimo or any environment
    where autoreload conflicts with torch internals.

    Returns the same dict as download_and_convert_to_onnx.
    """
    script_path = os.path.abspath(__file__)
    enc_flag = "None" if export_encoder_only is None else str(export_encoder_only)

    inner_code = (
        "import json, importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('_conv', r'{script_path}')\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "result = mod.download_and_convert_to_onnx(\n"
        f"    model_id=r'{model_id}',\n"
        f"    output_dir=r'{output_dir}',\n"
        f"    opset_version={opset_version},\n"
        f"    export_encoder_only={enc_flag},\n"
        f"    verify={verify},\n"
        ")\n"
        "print('__JSON_RESULT__' + json.dumps(result))\n"
    )

    proc = subprocess.run(
        [sys.executable, "-c", inner_code],
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Subprocess conversion failed (exit {proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )

    for line in proc.stdout.splitlines():
        if line.startswith("__JSON_RESULT__"):
            return json.loads(line[len("__JSON_RESULT__") :])

    raise RuntimeError(
        f"Could not find JSON result in subprocess output:\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )


# ====================================================================== #
#  Internal helpers
# ====================================================================== #


def _extract_encoder(model):
    """
    Return the encoder submodule from an encoder-decoder model.
    Works for Whisper, T5, BART, Marian, and similar HuggingFace models.
    """
    if hasattr(model, "get_encoder"):
        return model.get_encoder()
    if hasattr(model, "encoder"):
        return model.encoder
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        return model.model.encoder
    raise AttributeError(
        f"Cannot find encoder submodule on {type(model).__name__}. "
        "Pass export_encoder_only=False or provide a model_class that "
        "exposes .get_encoder() or .encoder."
    )


def _resolve_auto_class(config):
    """Pick the best AutoModel* class based on the config's architectures."""
    arch_map = {
        "WhisperForConditionalGeneration": AutoModelForSpeechSeq2Seq,
        "SpeechSeq2Seq": AutoModelForSpeechSeq2Seq,
        "ForConditionalGeneration": AutoModelForSeq2SeqLM,
        "Seq2SeqLM": AutoModelForSeq2SeqLM,
        "CausalLM": AutoModelForCausalLM,
        "SequenceClassification": AutoModelForSequenceClassification,
        "TokenClassification": AutoModelForTokenClassification,
        "AudioClassification": AutoModelForAudioClassification,
        "ImageClassification": AutoModelForImageClassification,
    }

    architectures = getattr(config, "architectures", None) or []
    for arch in architectures:
        for key, cls in arch_map.items():
            if key in arch:
                return cls
    return AutoModel


def _load_preprocessor(model_id: str):
    """Try loading a processor, tokenizer, or feature extractor — in that order."""
    for loader in [AutoProcessor, AutoTokenizer, AutoFeatureExtractor]:
        try:
            return loader.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            continue
    return None


def _build_dummy_input(config, preprocessor, is_enc_dec, export_encoder_only) -> dict:
    """Generate a dummy input dict suitable for the model architecture."""
    model_type = getattr(config, "model_type", "")

    # -------------------------------------------------------------- #
    # Audio / speech models (Whisper, Wav2Vec2, HuBERT, etc.)
    # -------------------------------------------------------------- #
    if model_type in ("whisper", "wav2vec2", "hubert", "sew", "unispeech"):
        sample_rate = getattr(config, "sampling_rate", 16000) or 16000
        dummy_audio = np.zeros(sample_rate * 2, dtype=np.float32)

        if preprocessor is not None:
            features = preprocessor(
                dummy_audio, sampling_rate=sample_rate, return_tensors="pt"
            )
            dummy = {k: v for k, v in features.items() if isinstance(v, torch.Tensor)}
        else:
            if model_type == "whisper":
                dummy = {"input_features": torch.randn(1, 80, 3000)}
            else:
                dummy = {"input_values": torch.randn(1, sample_rate * 2)}

        # Full encoder-decoder export needs decoder_input_ids
        if is_enc_dec and not export_encoder_only:
            dummy["decoder_input_ids"] = torch.tensor([[1, 1]], dtype=torch.long)

        return dummy

    # -------------------------------------------------------------- #
    # Vision models
    # -------------------------------------------------------------- #
    if model_type in ("vit", "deit", "beit", "swin", "convnext", "resnet", "clip"):
        if preprocessor is not None:
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            features = preprocessor(images=dummy_img, return_tensors="pt")
            return {k: v for k, v in features.items() if isinstance(v, torch.Tensor)}
        return {"pixel_values": torch.randn(1, 3, 224, 224)}

    # -------------------------------------------------------------- #
    # Text / default models
    # -------------------------------------------------------------- #
    if preprocessor is not None:
        try:
            features = preprocessor("Hello world", return_tensors="pt")
            dummy = {k: v for k, v in features.items() if isinstance(v, torch.Tensor)}
            if (
                is_enc_dec
                and not export_encoder_only
                and "decoder_input_ids" not in dummy
            ):
                dummy["decoder_input_ids"] = torch.tensor([[1, 1]], dtype=torch.long)
            return dummy
        except Exception:
            pass

    seq_len = min(getattr(config, "max_position_embeddings", 128), 128)
    dummy = {
        "input_ids": torch.ones(1, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
    }
    if is_enc_dec and not export_encoder_only:
        dummy["decoder_input_ids"] = torch.tensor([[1, 1]], dtype=torch.long)
    return dummy


def _resolve_output_names(model_output) -> list[str]:
    """Extract output tensor names from a model forward pass result."""
    if hasattr(model_output, "keys"):
        names = []
        for k in model_output.keys():
            v = model_output[k]
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                names.append(k)
            elif (
                isinstance(v, (tuple, list))
                and len(v) > 0
                and isinstance(v[0], torch.Tensor)
            ):
                names.append(k)
        return names if names else ["output_0"]
    if isinstance(model_output, (tuple, list)):
        return [f"output_{i}" for i in range(len(model_output))]
    return ["output_0"]


def _resolve_watsonx_spec(
    opset_version: int,
    software_spec_override: dict | None,
    hardware_spec_override: dict | None,
) -> tuple[dict, dict, str]:
    """
    Return (sw_spec, hw_spec, model_type) for watsonx.ai based on opset version.

    sw_spec and hw_spec are dicts with "name" and "id" keys.
    model_type is the string for ModelMetaNames.TYPE.
    """
    if opset_version <= 19:
        default_sw = {"name": "onnxruntime_opset_19", "id": None}
        model_type = "onnxruntime_1.16"
    else:
        default_sw = {
            "name": "onnxruntime_opset_21",
            "id": "d0b5361f-0810-554e-8d7b-874b5ff799b1",
        }
        model_type = "onnxruntime_1.17"

    default_hw = {"name": "L", "id": "a6c4923b-b8e4-444c-9f43-8a7ec3020110"}

    sw_spec = {**default_sw, **(software_spec_override or {})}
    hw_spec = {**default_hw, **(hardware_spec_override or {})}

    return sw_spec, hw_spec, model_type


# ONNX elem_type int → numpy dtype string mapping
_ONNX_TENSOR_TYPE_MAP = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    9: "bool",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
}


def _onnx_shape(tensor_type) -> list:
    """
    Extract shape from an ONNX TensorType, returning ints for fixed dims
    and -1 for dynamic dims.
    """
    shape = []
    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            shape.append(-1)
        else:
            shape.append(dim.dim_value if dim.dim_value > 0 else -1)
    return shape


def _build_model_schema(onnx_model) -> tuple[list, dict]:
    """
    Build watsonx.ai-compatible input and output schemas by reading the
    ONNX graph's input/output ValueInfoProto entries.

    Matches the actual ModelMetaNames contract:
      - INPUT_DATA_SCHEMA  → list  of {"id": str, "fields": [{"name": str, "type": str, "nullable": bool}]}
      - OUTPUT_DATA_SCHEMA → dict  of {"id": str, "fields": [{"name": str, "type": str, "nullable": bool}]}

    The "type" field encodes the ONNX tensor element type as a numpy-style
    dtype string (e.g. "float32", "int64") so that consumers know what data
    type to provide/expect when scoring.
    """
    input_fields = []
    for inp in onnx_model.graph.input:
        tt = inp.type.tensor_type
        dtype = _ONNX_TENSOR_TYPE_MAP.get(tt.elem_type, "float32")
        shape = _onnx_shape(tt)
        input_fields.append(
            {
                "name": inp.name,
                "type": f"tensor({dtype}){shape}",
                "nullable": False,
            }
        )

    output_fields = []
    for out in onnx_model.graph.output:
        tt = out.type.tensor_type
        dtype = _ONNX_TENSOR_TYPE_MAP.get(tt.elem_type, "float32")
        shape = _onnx_shape(tt)
        output_fields.append(
            {
                "name": out.name,
                "type": f"tensor({dtype}){shape}",
                "nullable": False,
            }
        )

    # INPUT_DATA_SCHEMA is a list of schema structs
    input_schema = [{"id": "input_schema", "fields": input_fields}]

    # OUTPUT_DATA_SCHEMA is a single schema dict
    output_schema = {"id": "output_schema", "fields": output_fields}

    return input_schema, output_schema


# ====================================================================== #
#  CLI / direct execution
# ====================================================================== #
if __name__ == "__main__":
    result = download_and_convert_to_onnx(
        model_id="openai/whisper-small",
        output_dir="./onnx_export",
        opset_version=21,
    )

    # To then upload to watsonx.ai Runtime:
    #
    # from ibm_watson_machine_learning import APIClient
    #
    # client = APIClient(your_credentials)
    #
    # # --- Store the model ---
    # meta_props = {
    #     client.repository.ModelMetaNames.NAME: "whisper-small-onnx",
    #     client.repository.ModelMetaNames.TYPE: result.get("model_type"),
    #     client.repository.ModelMetaNames.SOFTWARE_SPEC_ID: result.get("sw_spec", {}).get("id"),
    #     client.repository.ModelMetaNames.INPUT_DATA_SCHEMA: result.get("input_schema"),
    #     client.repository.ModelMetaNames.OUTPUT_DATA_SCHEMA: result.get("output_schema"),
    # }
    #
    # stored_model = client.repository.store_model(
    #     model=result.get("tar_gz_path"),
    #     meta_props=meta_props,
    # )
    # model_id = client.repository.get_model_id(stored_model)
    #
    # # --- Deploy (hardware_spec goes here, not on store_model) ---
    # deploy_props = {
    #     client.deployments.ConfigurationMetaNames.NAME: "whisper-small-onnx-deploy",
    #     client.deployments.ConfigurationMetaNames.ONLINE: {},
    #     client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: {
    #         "name": result.get("hw_spec", {}).get("name"),
    #         "id": result.get("hw_spec", {}).get("id"),
    #     },
    # }
    # deployment = client.deployments.create(model_id, meta_props=deploy_props)

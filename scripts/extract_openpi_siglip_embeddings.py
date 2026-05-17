#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_MODEL = "google/siglip-base-patch16-224"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract frozen SigLIP embeddings from OpenPI/LIBERO rollout videos")
    parser.add_argument("--config", help="Risk-training config with data.inputs")
    parser.add_argument("--input", action="append", default=[], help="Input JSONL path or glob; may be repeated")
    parser.add_argument("--output", default="outputs/openpi_libero/siglip_episode_embeddings.jsonl")
    parser.add_argument("--frame-dir", default="outputs/openpi_libero/siglip_frames")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dims", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--decoder-python", default="external/openpi/examples/libero/.venv/bin/python")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--extract-frames-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--requests", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.extract_frames_only:
        if not args.requests:
            raise SystemExit("--requests is required with --extract-frames-only")
        extract_requested_frames(Path(args.requests), force=args.force)
        return 0

    input_patterns = list(args.input)
    if args.config:
        input_patterns.extend(load_config_inputs(args.config))
    if not input_patterns:
        raise SystemExit("Provide --config or at least one --input")
    episodes = read_episodes(input_patterns)
    if args.limit is not None:
        episodes = episodes[: args.limit]
    if not episodes:
        raise SystemExit("No episodes found")

    output_path = Path(args.output)
    frame_dir = Path(args.frame_dir)
    requests_path = frame_dir / "frame_requests.jsonl"
    requests = build_frame_requests(episodes, frame_dir=frame_dir)
    requests_path.parent.mkdir(parents=True, exist_ok=True)
    requests_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in requests) + "\n", encoding="utf-8")
    run_frame_extraction(args.decoder_python, requests_path, force=args.force)

    existing = {} if args.force else read_existing_embeddings(output_path, model=args.model, dims=args.dims)
    rows = embed_frames(
        requests,
        existing=existing,
        model_name=args.model,
        dims=args.dims,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "episodes": len(episodes),
                "embeddings": len(rows),
                "model": args.model,
                "dims": args.dims,
                "output": str(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def load_config_inputs(path: str | Path) -> list[str]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data = config.get("data", {}) if isinstance(config, Mapping) else {}
    return [str(item) for item in data.get("inputs", [])]


def read_episodes(patterns: Sequence[str]) -> list[dict[str, Any]]:
    from risk_aware_skill_planning.risk.openpi_dataset import expand_input_paths

    episodes: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in expand_input_paths(patterns):
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                episode = json.loads(line)
                key = (
                    str(episode.get("run_id", episode.get("metadata", {}).get("run_id", ""))),
                    str(episode.get("episode_id", "")),
                )
                if key in seen:
                    continue
                seen.add(key)
                episodes.append(episode)
    return episodes


def build_frame_requests(episodes: Sequence[Mapping[str, Any]], *, frame_dir: Path) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for episode in episodes:
        run_id = str(episode.get("run_id", episode.get("metadata", {}).get("run_id", "")))
        episode_id = str(episode.get("episode_id", ""))
        video_path = str(episode.get("video_path", episode.get("metadata", {}).get("video_path", "")))
        if not run_id or not episode_id:
            raise ValueError(f"Episode is missing run_id or episode_id: {episode}")
        if not video_path:
            raise ValueError(f"Episode {run_id}/{episode_id} has no video_path")
        frame_path = frame_dir / f"{safe_name(run_id)}__{safe_name(episode_id)}.png"
        requests.append(
            {
                "run_id": run_id,
                "episode_id": episode_id,
                "suite": episode.get("libero_suite", episode.get("suite", "")),
                "task_id": episode.get("libero_task_id", episode.get("task_id", 0)),
                "stressor_name": episode.get("stressor_name", "none"),
                "success": bool(episode.get("success", False)),
                "video_path": video_path,
                "frame_path": str(frame_path),
            }
        )
    return requests


def run_frame_extraction(decoder_python: str, requests_path: Path, *, force: bool) -> None:
    decoder = Path(decoder_python)
    if not decoder.exists():
        raise FileNotFoundError(
            f"Decoder Python not found at {decoder}. Pass --decoder-python pointing at the OpenPI LIBERO venv."
        )
    command = [str(decoder), str(Path(__file__).resolve()), "--extract-frames-only", "--requests", str(requests_path)]
    if force:
        command.append("--force")
    subprocess.run(command, check=True)


def extract_requested_frames(requests_path: Path, *, force: bool) -> None:
    import imageio.v2 as imageio

    for row in read_jsonl(requests_path):
        frame_path = Path(row["frame_path"])
        if frame_path.exists() and not force:
            continue
        video_path = Path(row["video_path"])
        if not video_path.exists():
            raise FileNotFoundError(f"Missing rollout video: {video_path}")
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        reader = imageio.get_reader(str(video_path))
        try:
            image = reader.get_data(0)
        finally:
            reader.close()
        imageio.imwrite(str(frame_path), image)


def read_existing_embeddings(path: Path, *, model: str, dims: int) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    rows = {}
    for row in read_jsonl(path):
        if row.get("embedding_model") != model or int(row.get("embedding_dims", -1)) != dims:
            continue
        rows[(str(row.get("run_id", "")), str(row.get("episode_id", "")))] = row
    return rows


def embed_frames(
    requests: Sequence[Mapping[str, Any]],
    *,
    existing: Mapping[tuple[str, str], Mapping[str, Any]],
    model_name: str,
    dims: int,
    batch_size: int,
    device_name: str,
) -> list[dict[str, Any]]:
    from PIL import Image
    import torch
    from transformers import AutoModel, AutoProcessor

    device = choose_device(device_name, torch)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    rows: list[dict[str, Any]] = []
    pending: list[Mapping[str, Any]] = []
    for request in requests:
        key = (str(request["run_id"]), str(request["episode_id"]))
        if key in existing:
            rows.append(dict(existing[key]))
        else:
            pending.append(request)

    for batch in batched(pending, max(1, batch_size)):
        images = [Image.open(str(row["frame_path"])).convert("RGB") for row in batch]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            feature_output = model.get_image_features(**inputs)
            features = pooled_tensor(feature_output).float()
            features = torch.nn.functional.normalize(features, dim=-1).detach().cpu()
        for request, feature in zip(batch, features):
            compact = compact_embedding(feature.tolist(), dims)
            rows.append(
                {
                    "run_id": request["run_id"],
                    "episode_id": request["episode_id"],
                    "suite": request.get("suite"),
                    "task_id": request.get("task_id"),
                    "stressor_name": request.get("stressor_name"),
                    "success": request.get("success"),
                    "video_path": request["video_path"],
                    "frame_path": request["frame_path"],
                    "frame_source": "video_first_frame",
                    "embedding_model": model_name,
                    "embedding_dims": dims,
                    "embedding": compact,
                }
            )
    return sorted(rows, key=lambda row: (str(row["run_id"]), str(row["episode_id"])))


def choose_device(device_name: str, torch_module: Any) -> str:
    if device_name == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device_name


def pooled_tensor(output: Any) -> Any:
    if hasattr(output, "float"):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state.mean(dim=1)
    raise TypeError(f"Unsupported image feature output type: {type(output)!r}")


def compact_embedding(values: Sequence[float], dims: int) -> list[float]:
    if dims <= 0:
        raise ValueError("dims must be positive")
    if dims >= len(values):
        return [float(value) for value in values]
    compact: list[float] = []
    width = len(values) / dims
    for idx in range(dims):
        start = int(math.floor(idx * width))
        stop = int(math.floor((idx + 1) * width))
        stop = max(stop, start + 1)
        bucket = values[start:stop]
        compact.append(float(sum(bucket) / len(bucket)))
    return compact


def batched(items: Sequence[Mapping[str, Any]], batch_size: int) -> Iterable[Sequence[Mapping[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())

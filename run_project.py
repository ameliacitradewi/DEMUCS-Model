from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from demucs_project.config import ProjectConfig, load_config
from demucs_project.evaluation import evaluate_dataset, evaluate_track
from demucs_project.inference import separate_audio
from demucs_project.paths import (
    find_track_directory,
    list_track_directories,
    safe_external_name,
)
from demucs_project.reporting import build_report
from demucs_project.validation import validate_dataset
from demucs_project.visualization import visualize_track


def _print_json(value) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def _resolve_track(config: ProjectConfig, split: str, track: str | None) -> Path:
    if track:
        return find_track_directory(config.dataset_root, split, track)
    tracks = list_track_directories(config.dataset_root, split)
    if not tracks:
        raise RuntimeError(f"Tidak ada track pada split '{split}'.")
    print(f"--track tidak diberikan; menggunakan track pertama: {tracks[0].name}")
    return tracks[0]


def command_validate(args, config: ProjectConfig) -> None:
    splits = tuple(item.strip() for item in args.splits.split(",") if item.strip())
    frame, summary = validate_dataset(
        config,
        splits=splits,
        check_mixture_sum=args.check_mixture_sum,
        sum_check_limit=args.sum_check_limit,
    )
    print(frame[["split", "track", "status", "issues"]].to_string(index=False))
    print("\nRingkasan:")
    _print_json(summary)


def command_list_tracks(args, config: ProjectConfig) -> None:
    tracks = list_track_directories(config.dataset_root, args.split)
    for index, track in enumerate(tracks, start=1):
        print(f"{index:03d}. {track.name}")
    print(f"\nTotal: {len(tracks)} track")


def command_separate_one(args, config: ProjectConfig) -> None:
    track_dir = _resolve_track(config, args.split, args.track)
    result = separate_audio(
        config,
        input_audio=track_dir / "mixture.wav",
        output_name=track_dir.name,
        overwrite=args.overwrite,
    )
    _print_json(result)


def command_separate_dataset(args, config: ProjectConfig) -> None:
    tracks = list_track_directories(config.dataset_root, args.split)
    if args.limit is not None:
        tracks = tracks[: args.limit]
    failures = []
    for index, track_dir in enumerate(tracks, start=1):
        print(f"\n[{index}/{len(tracks)}] Inferensi: {track_dir.name}")
        try:
            result = separate_audio(
                config,
                input_audio=track_dir / "mixture.wav",
                output_name=track_dir.name,
                overwrite=args.overwrite,
            )
            print(f"Status: {result['status']} | output: {result['output']}")
        except Exception as exc:
            failures.append(
                {
                    "track": track_dir.name,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            print(f"Gagal: {type(exc).__name__}: {exc}")
            if not args.continue_on_error:
                raise

    print(f"\nSelesai. Berhasil/skip: {len(tracks) - len(failures)}, gagal: {len(failures)}")
    if failures:
        failure_path = config.reports_root / f"inference_failures_{args.split}.csv"
        pd.DataFrame(failures).to_csv(failure_path, index=False)
        print(f"Daftar kegagalan: {failure_path}")


def command_evaluate_one(args, config: ProjectConfig) -> None:
    track_dir = _resolve_track(config, args.split, args.track)
    frame, summary = evaluate_track(
        config,
        split=args.split,
        track_name=track_dir.name,
        with_bss_eval=args.bss,
    )
    columns = [
        "stem",
        "si_sdr_db",
        "si_sdri_db",
        "waveform_sdr_db",
        "spectral_convergence",
        "log_spectral_mae_db",
    ]
    if args.bss:
        columns.extend(["bss_sdr_db", "bss_isr_db", "bss_sir_db", "bss_sar_db"])
    print(frame[columns].round(4).to_string(index=False))
    print("\nTrack summary:")
    _print_json(summary)


def command_evaluate_dataset(args, config: ProjectConfig) -> None:
    metrics, summaries, errors = evaluate_dataset(
        config,
        split=args.split,
        with_bss_eval=args.bss,
        limit=args.limit,
        continue_on_error=args.continue_on_error,
    )
    print(
        f"\nTrack berhasil: {summaries.shape[0]} | "
        f"baris metrik: {metrics.shape[0]} | gagal: {errors.shape[0]}"
    )
    if not metrics.empty:
        columns = ["stem", "si_sdr_db", "si_sdri_db"]
        if "bss_sdr_db" in metrics.columns:
            columns.append("bss_sdr_db")
        print("\nMedian per stem:")
        print(metrics.groupby("stem")[columns[1:]].median().round(4).to_string())


def command_visualize_one(args, config: ProjectConfig) -> None:
    track_dir = _resolve_track(config, args.split, args.track)
    paths = visualize_track(
        config,
        split=args.split,
        track_name=track_dir.name,
        start_seconds=args.start,
        duration_seconds=args.duration,
        include_extras=args.extras,
    )
    print("File visualisasi dibuat:")
    for path in paths:
        print(f"- {path}")


def command_pipeline_one(args, config: ProjectConfig) -> None:
    track_dir = _resolve_track(config, args.split, args.track)
    print("\n1/3 Inferensi")
    _print_json(
        separate_audio(
            config,
            input_audio=track_dir / "mixture.wav",
            output_name=track_dir.name,
            overwrite=args.overwrite,
        )
    )

    print("\n2/3 Evaluasi")
    frame, summary = evaluate_track(
        config,
        split=args.split,
        track_name=track_dir.name,
        with_bss_eval=args.bss,
    )
    display_columns = ["stem", "si_sdr_db", "si_sdri_db", "spectral_convergence"]
    if args.bss:
        display_columns.append("bss_sdr_db")
    print(frame[display_columns].round(4).to_string(index=False))
    _print_json(summary)

    print("\n3/3 Visualisasi")
    paths = visualize_track(
        config,
        split=args.split,
        track_name=track_dir.name,
        start_seconds=args.start,
        duration_seconds=args.duration,
        include_extras=args.extras,
    )
    for path in paths:
        print(f"- {path}")


def command_report(args, config: ProjectConfig) -> None:
    outputs = build_report(config, split=args.split)
    print("Report dibuat:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


def command_external(args, config: ProjectConfig) -> None:
    input_audio = Path(args.audio).expanduser().resolve()
    output_name = args.name or safe_external_name(input_audio)
    result = separate_audio(
        config,
        input_audio=input_audio,
        output_name=output_name,
        overwrite=args.overwrite,
    )
    _print_json(result)

    if args.visualize:
        print("\nMembuat visualisasi untuk lagu eksternal...")
        separated_dir = Path(result["output"])
        from demucs_project.visualization import visualize_external_track
        visualizations = visualize_external_track(
            config,
            separated_dir=separated_dir,
            output_name=output_name,
            start_seconds=args.start,
            duration_seconds=args.duration,
        )
        print("Visualisasi berhasil dibuat:")
        for path in visualizations:
            print(f"- {path}")

    print(
        "\nCatatan: tanpa ground-truth stems, SDR/SI-SDR/SIR/SAR tidak dapat "
        "dihitung secara valid. Gunakan listening test dan pemeriksaan artefak."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline Demucs V4 + evaluasi MUSDB18-HQ",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path file YAML konfigurasi (default: config.yaml)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validasi struktur dataset")
    validate.add_argument("--splits", default="train,valid,test")
    validate.add_argument("--check-mixture-sum", action="store_true")
    validate.add_argument("--sum-check-limit", type=int)
    validate.set_defaults(function=command_validate)

    list_tracks = subparsers.add_parser("list-tracks", help="Tampilkan nama track")
    list_tracks.add_argument("--split", default="test")
    list_tracks.set_defaults(function=command_list_tracks)

    separate_one = subparsers.add_parser("separate-one", help="Inferensi satu track")
    separate_one.add_argument("--split", default="test")
    separate_one.add_argument("--track")
    separate_one.add_argument("--overwrite", action="store_true")
    separate_one.set_defaults(function=command_separate_one)

    separate_dataset = subparsers.add_parser(
        "separate-dataset", help="Inferensi seluruh track dalam split"
    )
    separate_dataset.add_argument("--split", default="test")
    separate_dataset.add_argument("--limit", type=int)
    separate_dataset.add_argument("--overwrite", action="store_true")
    separate_dataset.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    separate_dataset.set_defaults(function=command_separate_dataset)

    evaluate_one = subparsers.add_parser("evaluate-one", help="Evaluasi satu track")
    evaluate_one.add_argument("--split", default="test")
    evaluate_one.add_argument("--track")
    evaluate_one.add_argument("--bss", action="store_true")
    evaluate_one.set_defaults(function=command_evaluate_one)

    evaluate_dataset_parser = subparsers.add_parser(
        "evaluate-dataset", help="Evaluasi seluruh track dalam split"
    )
    evaluate_dataset_parser.add_argument("--split", default="test")
    evaluate_dataset_parser.add_argument("--limit", type=int)
    evaluate_dataset_parser.add_argument("--bss", action="store_true")
    evaluate_dataset_parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    evaluate_dataset_parser.set_defaults(function=command_evaluate_dataset)

    visualize = subparsers.add_parser("visualize-one", help="Visualisasi satu track")
    visualize.add_argument("--split", default="test")
    visualize.add_argument("--track")
    visualize.add_argument("--start", type=float)
    visualize.add_argument("--duration", type=float)
    visualize.add_argument("--extras", action="store_true")
    visualize.set_defaults(function=command_visualize_one)

    pipeline = subparsers.add_parser(
        "pipeline-one", help="Inferensi + evaluasi + visualisasi satu track"
    )
    pipeline.add_argument("--split", default="test")
    pipeline.add_argument("--track")
    pipeline.add_argument("--bss", action="store_true")
    pipeline.add_argument("--start", type=float)
    pipeline.add_argument("--duration", type=float)
    pipeline.add_argument("--extras", action="store_true")
    pipeline.add_argument("--overwrite", action="store_true")
    pipeline.set_defaults(function=command_pipeline_one)

    report = subparsers.add_parser("report", help="Agregasi hasil evaluasi")
    report.add_argument("--split", default="test")
    report.set_defaults(function=command_report)

    external = subparsers.add_parser("external", help="Pisahkan lagu eksternal")
    external.add_argument("--audio", required=True)
    external.add_argument("--name")
    external.add_argument("--overwrite", action="store_true")
    external.add_argument("--visualize", action="store_true", help="Buat visualisasi waveform dan spectrogram untuk setiap stem")
    external.add_argument("--start", type=float, help="Detik mulai untuk visualisasi (default dari config)")
    external.add_argument("--duration", type=float, help="Durasi visualisasi dalam detik (default dari config)")
    external.set_defaults(function=command_external)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        args.function(args, config)
        return 0
    except KeyboardInterrupt:
        print("\nDibatalkan oleh pengguna.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

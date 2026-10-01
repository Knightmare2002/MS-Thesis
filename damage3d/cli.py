"""Command-line interface: python -m damage3d [options]  (see damage3d/README.md)."""

from __future__ import annotations

import argparse
import logging
import sys

from .paths import DEFAULT_NAMES, ENV_PROJECT_ROOT, resolve_path, resolve_project_root
from .pipeline import RunConfig, run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="damage3d",
        description="Project multilabel 2D damage probabilities onto the Metashape point cloud "
                    "(projection, mesh occlusion, multi-view fusion, PLY/NPZ export).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("paths (relative paths are resolved against --project-root)")
    g.add_argument("--project-root", help=f"bridge_model folder; default ${ENV_PROJECT_ROOT} or the thesis path")
    g.add_argument("--images-dir", help=f"photos folder (optional size/EXIF check); default '{DEFAULT_NAMES['images_dir']}'")
    g.add_argument("--cameras-xml", help=f"default '{DEFAULT_NAMES['cameras_xml']}'")
    g.add_argument("--point-cloud", help=f"default '{DEFAULT_NAMES['point_cloud']}'")
    g.add_argument("--mesh", help=f"default '{DEFAULT_NAMES['mesh']}'")
    g.add_argument("--marker-reference", help=f"default '{DEFAULT_NAMES['marker_reference']}' (frame check)")
    g.add_argument("--output-dir", help="run directory; default <project-root>/damage3d_runs/<auto name>")

    g = p.add_argument_group("probabilities")
    g.add_argument("--probability-source", choices=("synthetic", "files"), default="synthetic")
    g.add_argument("--probabilities-dir", help="folder with <stem>.npz maps (files provider)")
    g.add_argument("--synthetic-pattern", choices=("regions", "mask"), default="regions",
                   help="'regions': six overlapping Gaussian bumps; 'mask': prototype PNG mask")
    g.add_argument("--synthetic-downsample", type=int, default=8, help="resolution divisor of synthetic maps")
    g.add_argument("--synthetic-mask-pattern", default="{stem}_mask_test.png",
                   help="mask file name in the project root, for --synthetic-pattern mask")
    g.add_argument("--synthetic-mask-classes", default="crack", help="comma-separated classes set to 1 on white pixels")

    g = p.add_argument_group("camera selection (end is EXCLUSIVE; name and range selectors are mutually exclusive)")
    g.add_argument("--camera", action="append", help="camera label or stem; repeatable")
    g.add_argument("--camera-list", help="text file, one label/stem per line, '#' comments")
    g.add_argument("--start-image", type=int, help="first position (0-based) in the sorted aligned cameras")
    g.add_argument("--end-image", type=int, help="position AFTER the last selected camera (exclusive)")
    g.add_argument("--limit-images", type=int, help="keep only the first N selected cameras (applied last)")

    g = p.add_argument_group("points and memory")
    g.add_argument("--max-points", type=int, help="reproducible subset of the PLY (default: all points)")
    g.add_argument("--sample-mode", choices=("random", "stride"), default="random")
    g.add_argument("--seed", type=int, default=0, help="seed of --sample-mode random")
    g.add_argument("--point-chunk-size", type=int, default=2_000_000)
    g.add_argument("--camera-batch-size", type=int, default=4, help="cameras per commit; maps kept in RAM per batch")

    g = p.add_argument_group("geometry")
    g.add_argument("--point-frame", choices=("auto", "chunk", "transformed"), default="auto",
                   help="frame of PLY/OBJ w.r.t. the Metashape chunk (see damage3d/frames.py)")
    g.add_argument("--occlusion", choices=("mesh", "none"), default="mesh")
    g.add_argument("--occlusion-tolerance", type=float, help="absolute tolerance (source units); default auto")
    g.add_argument("--occlusion-rel-tolerance", type=float, default=0.0, help="extra tolerance per unit distance")
    g.add_argument("--interpolation", choices=("bilinear", "nearest"), default="bilinear")
    g.add_argument("--border-margin-px", type=float, default=0.0, help="discard projections closer to the border")
    g.add_argument("--min-depth", type=float, default=0.0, help="camera Z must be > this value")

    g = p.add_argument_group("fusion and export (can change on --resume: outputs are re-exported)")
    g.add_argument("--vote-threshold", type=float, default=0.5, help="per-view threshold for the votes counters")
    g.add_argument("--thresholds", default="0.5",
                   help="'0.5' or 'crack=0.5,spalling=0.5,...'; synthetic runs: declared TEST thresholds")
    g.add_argument("--min-views", type=int, default=1)
    g.add_argument("--export-scope", choices=("all", "observed", "damage"), default="all")
    g.add_argument("--csv-max-points", type=int, default=200_000)

    g = p.add_argument_group("control")
    g.add_argument("--dry-run", action="store_true", help="validate inputs and print the plan; write nothing")
    g.add_argument("--resume", action="store_true", help="continue an interrupted run in --output-dir")
    g.add_argument("--verbose", action="store_true")
    g.add_argument("--test-interrupt-after-writes", type=int, help=argparse.SUPPRESS)
    return p


def config_from_args(args: argparse.Namespace) -> RunConfig:
    root = resolve_project_root(args.project_root)
    occlusion_mesh = args.occlusion == "mesh"
    return RunConfig(
        project_root=root,
        images_dir=resolve_path(root, args.images_dir, "images_dir"),
        cameras_xml=resolve_path(root, args.cameras_xml, "cameras_xml"),
        point_cloud=resolve_path(root, args.point_cloud, "point_cloud"),
        mesh=resolve_path(root, args.mesh, "mesh") if occlusion_mesh or args.mesh else None,
        marker_reference=resolve_path(root, args.marker_reference, "marker_reference"),
        output_dir=resolve_path(root, args.output_dir, None),
        runs_dir=resolve_path(root, None, "runs_dir"),
        probability_source=args.probability_source,
        probabilities_dir=resolve_path(root, args.probabilities_dir, None),
        synthetic_pattern=args.synthetic_pattern,
        synthetic_downsample=args.synthetic_downsample,
        synthetic_mask_pattern=args.synthetic_mask_pattern,
        synthetic_mask_classes=tuple(c.strip() for c in args.synthetic_mask_classes.split(",") if c.strip()),
        camera=args.camera,
        camera_list=resolve_path(root, args.camera_list, None),
        start_image=args.start_image,
        end_image=args.end_image,
        limit_images=args.limit_images,
        max_points=args.max_points,
        sample_mode=args.sample_mode,
        seed=args.seed,
        point_chunk_size=args.point_chunk_size,
        camera_batch_size=args.camera_batch_size,
        point_frame=args.point_frame,
        occlusion=args.occlusion,
        occlusion_tolerance=args.occlusion_tolerance,
        occlusion_rel_tolerance=args.occlusion_rel_tolerance,
        interpolation=args.interpolation,
        border_margin_px=args.border_margin_px,
        min_depth=args.min_depth,
        vote_threshold=args.vote_threshold,
        thresholds=args.thresholds,
        min_views=args.min_views,
        export_scope=args.export_scope,
        csv_max_points=args.csv_max_points,
        dry_run=args.dry_run,
        resume=args.resume,
        test_interrupt_after_writes=args.test_interrupt_after_writes,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = logging.getLogger("damage3d")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    if args.point_chunk_size <= 0 or args.camera_batch_size <= 0:
        raise SystemExit("--point-chunk-size and --camera-batch-size must be > 0.")
    run(config_from_args(args))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

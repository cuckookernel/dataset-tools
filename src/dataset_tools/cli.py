import datetime as dt
from pathlib import Path
from typing import Annotated, Sequence, TypeAlias, TypedDict
from dataset_tools.output_manager import OutputManager, FmtSpec

import polars as pl
import typer

InputDataArgument: TypeAlias = Annotated[
    Path, typer.Argument(help="Path to input data file or directory")
]

OutputCliOptionMaybe: TypeAlias = Annotated[
    Path | None,
    typer.Option(..., "-o", "--out", help="Path to output data file or directory"),
]

OutputCliOptionRequired: TypeAlias = Annotated[
    Path, typer.Option(..., "-o", "--out", help="Path to output data file or directory")
]


FormatCliOpt: TypeAlias = Annotated[FmtSpec, typer.Option(help="output format")]

ExtensionCliOpt: TypeAlias = Annotated[
    str, typer.Option(help="File extension spec, e.g 'parquet', 'pq', 'csv'")
]

cli = typer.Typer(no_args_is_help=True)


def input_paths_spec(path: Path, ext) -> str:
    if path.is_dir():
        return f"{path}/*.parquet"
    else:
        return str(path)


@cli.command()
def count(
    input: InputDataArgument,
    ext: ExtensionCliOpt = "parquet",
    fmt: FormatCliOpt = FmtSpec.HUMAN,
):
    input_path1 = input_paths_spec(input, ext=ext)
    del input

    input_lf = pl.scan_parquet(input_path1)
    cnt: int = input_lf.select(pl.len()).collect().item()
    OutputManager(fmt=fmt, out_path=None).emit(cnt)


@cli.command()
def schema(
    input: InputDataArgument,
    ext: ExtensionCliOpt = "parquet",
    fmt: FormatCliOpt = FmtSpec.HUMAN,
):
    input_path1 = input_paths_spec(input, ext=ext)
    del input

    input_lf = pl.scan_parquet(input_path1)
    sch: pl.Schema = input_lf.collect_schema()
    OutputManager(fmt=fmt, out_path=None).emit(sch)


@cli.command()
def count_by_file(
    input: InputDataArgument,
    out_path: OutputCliOptionMaybe = None,
    ext: ExtensionCliOpt = "parquet",
    fmt: FormatCliOpt = FmtSpec.HUMAN,
):
    if input.is_dir():
        files = list(input.rglob(f"*.{ext}"))
    else:
        files = [input]

    class CountRec(TypedDict):
        idx: int
        path: str
        num_rows: int

    count_by_file: Sequence[CountRec] = []

    for i, file in enumerate(sorted(files)):
        cnt: int = pl.scan_parquet(file).select(pl.len()).collect().item()
        count_by_file.append({"idx": i, "path": str(file), "num_rows": cnt})

    total_files = len(count_by_file)
    total_num_rows: int = sum(d["num_rows"] for d in count_by_file)

    out_man = OutputManager(fmt=fmt, out_path=out_path)
    out_man.emit_df_and_global_stats(
        df=pl.DataFrame(count_by_file),
        global_stats={
            "total_files": total_files,
            "total_num_rows": total_num_rows,
        },
    )


@cli.command()
def stats_by_file(
    input: InputDataArgument,
    out_path: OutputCliOptionMaybe = None,
    ext: ExtensionCliOpt = "parquet",
    fmt: FormatCliOpt = FmtSpec.HUMAN,
):
    if input.is_dir():
        files = list(input.rglob(f"*.{ext}"))
    else:
        files = [input]

    class StatsRec(TypedDict):
        idx: int
        path: str
        size_mb: float
        num_rows: int
        last_modified: dt.datetime

    stats_by_file: Sequence[StatsRec] = []

    for i, file in enumerate(sorted(files)):
        cnt: int = pl.scan_parquet(file).select(pl.len()).collect().item()
        stats_by_file.append(
            {
                "idx": i,
                "path": str(file),
                "num_rows": cnt,
                "size_mb": file.lstat().st_size / 1e6,
                "last_modified": dt.datetime.fromtimestamp(
                    file.lstat().st_mtime, tz=dt.UTC
                ).isoformat(),
            }
        )

    stats_df = pl.DataFrame(stats_by_file)

    global_stats = {
        "total_files": len(stats_df),
        "total_num_rows": stats_df["num_rows"].sum(),
        "total_size_mb": stats_df["size_mb"].sum(),
        "last_modified": stats_df["last_modified"].max(),
    }

    global_stats["avg_bytes_by_row"] = (
        global_stats["total_size_mb"]
        * 1e6  # ty:ignore[unsupported-operator]
        / global_stats["total_num_rows"]
    )

    out_man = OutputManager(fmt=fmt, out_path=out_path)
    out_man.emit_df_and_global_stats(
        pl.DataFrame(stats_by_file),
        global_stats=global_stats,  # ty: ignore[invalid-argument-type]
    )


if __name__ == "__main__":
    cli()

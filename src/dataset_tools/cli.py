import datetime as dt
import io
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Mapping, Sequence, TypeAlias, TypedDict

import polars as pl
import typer


class FmtSpec(StrEnum):
    HUMAN = "human"
    PLAIN = "plain"
    JSON = "json"  # single line json, no line breaks
    JSONL = "jsonl"  # works for an array, one line per item in array
    JSONH = "jsonh"  # indented json for human consumption
    MARKDOWN = "md"
    CSV = "csv"


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


@dataclass
class HumanFmtLines:
    lines: list[str]


@dataclass
class OutputManager:
    fmt: FmtSpec
    out_path: Path | None
    json_indent: int = 2
    float_precision: int = 6
    fmt_max_rows: int | None = None
    fmt_max_col_width: int | None = None

    # Output formatting functions alway return `str``
    def format_int(self, i: int) -> str:
        match self.fmt:
            case "human":
                return f"{i:,d}"
            case "plain":
                return str(i)
            case "json" | "jsonl" | "jsonh":
                return json.dumps(i)
            case _:
                raise ValueError(f"Unhandled format `{self.fmt}` for int")

    def format_schema(self, sch: pl.Schema) -> str:
        dict_list = [
            {"column": name, "dtype": str(dtype)}
            for name, dtype in zip(sch.names(), sch.dtypes(), strict=True)
        ]
        match self.fmt:
            case "plain":
                return str(sch)
            case "human":
                lines = [
                    f"{name}: {dtype}"
                    for name, dtype in zip(sch.names(), sch.dtypes(), strict=True)
                ]
                lines.append(f"\nTOTAL: {len(sch.names())} columns")
                return "\n".join(lines)
            case "json" | "jsonh":
                return self.format_json(
                    {"num_columns": len(dict_list), "columns": dict_list}
                )
            case "jsonl":
                return self.format_list_json(dict_list)
            case _:
                raise ValueError(f"Don't know how to format with {self.fmt=}")

    def format_json(self, obj: Any) -> str:
        match self.fmt:
            case "json" | "jsonh":
                indent = self.json_indent if self.fmt == "jsonh" else None
                return json.dumps(obj, indent=indent)
            case _:
                raise ValueError(
                    "Only valid value for fmt are 'json', 'jsonl', 'jsonh'"
                )

    def format_list_json(self, dict_list: Sequence[Mapping[str, Any]]) -> str:
        match self.fmt:
            case "json" | "jsonh":
                return self.format_json(dict_list)
            case "jsonl":
                return "\n".join(json.dumps(d) for d in dict_list)
            case _:
                raise ValueError(
                    "Only valid value for fmt are 'json', 'jsonl', 'jsonh'"
                )

    def format_df(self, df: pl.DataFrame) -> str:
        match self.fmt:
            case "plain":
                return str(df)
            case "json" | "jsonh":
                return self.format_json(df.to_dicts())
            case "jsonl":
                return self.format_list_json(df.to_dicts())
            case "csv":
                mem_file = io.BytesIO()
                df.to_pandas().to_csv(mem_file)
                mem_file.seek(0)
                return mem_file.getvalue().decode("utf-8")
            case "human" | "md":
                import pandas as pd

                mem_file = io.StringIO()
                with pd.option_context(
                    "display.max_rows",
                    self.fmt_max_rows,
                    "display.max_colwidth",
                    self.fmt_max_col_width,
                ):
                    df.to_pandas().to_markdown(mem_file, index=False)
                mem_file.seek(0)
                return mem_file.getvalue()
            case _:
                raise ValueError(f"Don't know how to format with {self.fmt=}")

    def format(self, obj: Any) -> str:
        if isinstance(obj, int):
            return self.format_int(obj)
        elif isinstance(obj, float):
            return f"{obj:.{self.float_precision}}"
        elif isinstance(obj, str):
            return obj
        elif isinstance(obj, pl.DataFrame):
            return self.format_df(obj)
        elif isinstance(obj, dict):
            return self.format_json(obj)
        elif isinstance(obj, list):
            return self.format_list_json(obj)
        elif isinstance(obj, pl.Schema):
            return self.format_schema(obj)
        elif isinstance(obj, HumanFmtLines):
            return "\n".join(obj.lines)
        else:
            raise TypeError(f"don't know how to handle object of type {type(obj)}")

    def save(self, obj: Any) -> None:
        out_path = self.out_path
        fmt = self.fmt

        if out_path is None:
            raise ValueError("I should not be called if out_path is None")

        if isinstance(obj, pl.DataFrame):
            match fmt:
                case "parquet":
                    obj.write_parquet(out_path)
                case "json" | "jsonh":
                    obj_out = {
                        "num_rows": len(obj),
                        "num_columns": len(obj.columns),
                        "columns": [str(col) for col in obj.columns],
                        "dtypes": [str(dtype) for dtype in obj.schema.dtypes],
                        "rows": obj.to_dicts(),
                    }
                    with out_path.open("wt") as f_out:
                        indent = 2 if fmt == "jsonh" else None
                        json.dump(obj_out, f_out, indent=indent)
                case "jsonl":
                    obj.write_ndjson(out_path)
                case "csv":
                    obj.write_csv(out_path)
                case "md":
                    obj.to_pandas().to_markdown(out_path, index=False)
        elif isinstance(obj, pl.LazyFrame):
            raise TypeError("LazyFrame")
        else:
            out_str = self.format(obj)
            with out_path.open("wt") as f_out:
                f_out.write(out_str)

    def emit(self, obj: Any) -> None:
        """Either save to out if Path is not None, or print formatted value to stdout"""

        if self.out_path is not None:
            self.save(obj)
        else:
            print(self.format(obj))

    def emit_df_and_global_stats(
        self, df: pl.DataFrame, global_stats: Mapping[str, str | int | float]
    ) -> None:
        match self.fmt:
            case "jsonl":
                self.emit(df)
            case "json" | "jsonh":
                out_obj = {"items": df.to_dicts(), **global_stats}
                self.emit(out_obj)
            case "human" | "plain":
                lines = self.format(df)
                key_width = max(len(key) for key in global_stats.keys()) + 1
                out_obj = HumanFmtLines(
                    lines=[
                        lines,
                        "---",
                    ]
                    + [
                        f"{key:{key_width}s}: {self.format(value)}"
                        for key, value in global_stats.items()
                    ]
                )
                return self.emit(out_obj)
            case "md" | "csv":
                self.emit(df)
            case _:
                raise ValueError(f"fmt={self.fmt} not handled yet")


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

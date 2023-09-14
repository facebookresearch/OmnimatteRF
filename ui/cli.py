# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import sys
from argparse import ArgumentParser
from inspect import Parameter, signature

from ui.commands import all_cli_commands
from ui.common import create_data_manager
from ui.data_manager import DataManager


def main():
    argv = sys.argv

    logging.basicConfig(
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
        level=logging.INFO,
    )

    if len(argv) < 2 or argv[1] not in all_cli_commands:
        cmd_names = sorted(all_cli_commands.keys())
        print(f"Usage: {argv[0]} command [...args]")
        print(f"Commands:")
        print("\n".join(["  " + cmd for cmd in cmd_names]))
        exit(1)

    func = all_cli_commands[argv[1]]
    sig = signature(func)
    dm_param = None
    has_extra_args = False

    parser = ArgumentParser(argv[1])
    for param in sig.parameters.values():
        if param.annotation == DataManager:
            dm_param = param.name
            continue
        if param.name == "extra_args":
            has_extra_args = True
            continue
        if param.annotation == list[str]:
            parser.add_argument(f"--{param.name}", type=str, default=param.default, action="append")
            continue

        if param.annotation == bool:
            assert param.default in { True, False }, f"bool param ({param.name}) must have default"
            if not param.default:
                parser.add_argument(f"--{param.name}", action="store_true")
            else:
                parser.add_argument(f"--no_{param.name}", dest=param.name, action="store_false")
            continue

        if param.default != Parameter.empty:
            parser.add_argument(f"--{param.name}", type=param.annotation, default=param.default)
        else:
            parser.add_argument(f"{param.name}", type=param.annotation)

    argv = argv[2:]
    extra_args = []
    if has_extra_args and "--" in argv:
        split_idx = argv.index("--")
        extra_args = argv[split_idx+1:]
        argv = argv[:split_idx]

    args = parser.parse_args(argv)
    args = vars(args)
    if dm_param is not None:
        dm = create_data_manager()
        args[dm_param] = dm
    if has_extra_args:
        args["extra_args"] = extra_args

    func(**args)


if __name__ == "__main__":
    main()

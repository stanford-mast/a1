"""
Command-line interface for a1.

Provides utilities for:
- Inspecting agents and tools
- Running agents from definitions
- Clearing cache
"""

import argparse
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="a1 - Agent compiler CLI", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--version", action="store_true", help="Show version and exit")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Cache command
    cache_parser = subparsers.add_parser("cache", help="Manage compilation cache")
    cache_parser.add_argument("action", choices=["clear", "list"], help="Action to perform on cache")
    cache_parser.add_argument("--dir", default=".a1", help="Cache directory (default: .a1)")

    args = parser.parse_args()

    if args.version:
        from . import __version__

        print(f"a1 version {__version__}")
        return

    if args.command == "cache":
        handle_cache(args)
    else:
        parser.print_help()


def handle_cache(args):
    """Handle cache management commands."""
    cache_dir = Path(args.dir)

    if args.action == "clear":
        if cache_dir.exists():
            count = 0
            for file in cache_dir.glob("*.py"):
                file.unlink()
                count += 1
            print(f"Cleared {count} cached files from {cache_dir}")
        else:
            print(f"Cache directory {cache_dir} does not exist")

    elif args.action == "list":
        if cache_dir.exists():
            files = list(cache_dir.glob("*.py"))
            if files:
                print(f"Cached files in {cache_dir}:")
                for file in files:
                    size = file.stat().st_size
                    print(f"  {file.name} ({size} bytes)")
            else:
                print(f"No cached files in {cache_dir}")
        else:
            print(f"Cache directory {cache_dir} does not exist")


if __name__ == "__main__":
    main()

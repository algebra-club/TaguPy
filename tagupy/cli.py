"""Console script for tagupy."""
import sys
from datetime import datetime

import click
from cookiecutter.main import cookiecutter

import tagupy


@click.group()
def main():
    return 0


@main.command(help='Create New Project')
@click.option(
    '--name', '-n',
    prompt='project name?',
    help='New Project Name')
def new(name):
    cookiecutter(
        f'{list(tagupy.__path__)[0]}/template',
        no_input=True,
        extra_context={
            "prj_name": name,
            "current_time": datetime.now().strftime('%b-%d-%Y'),
        }
    )


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

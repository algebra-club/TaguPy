"""Console script for tagupy."""
import os
import sys
from datetime import datetime

import click
from cookiecutter.main import cookiecutter

import tagupy


@click.group()
def main():
    return 0


def is_project(path: str) -> bool:
    return os.path.isfile(f'{path}/config.toml')


@main.command(help='Create New Project')
@click.option(
    '--name', '-n',
    help='New Project Name',
    prompt='project name?',
    required=True,
)
def new(name):
    cookiecutter(
        f'{list(tagupy.__path__)[0]}/template',
        no_input=True,
        extra_context={
            "prj_name": name,
            "current_time": datetime.now().strftime('%b-%d-%Y'),
        }
    )


@main.command(help='Generate Experience Matrix')
@click.option(
    '--name', '-n',
    help='New Experience Name',
    prompt='experience name?',
    required=True,
)
@click.option(
    '--method', '-m',
    prompt='generating method?',
    help='Generate method of experience matrix',
    type=click.Choice(tagupy.design.generator.__all__, case_sensitive=False),
)
def generate(name, method):
    if not is_project(os.getcwd()):
        click.echo('You need to locate on project root. Terminated')
        return
    from tagupy.manager import generator
    method = getattr(generator, method)
    params = method.required_params()
    res = {}
    for prompt, (k, t) in params.items():
        res[k] = click.prompt(prompt, type=t)

    exmatrix = method.generate(**res)
    click.echo(name)
    click.echo(exmatrix)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

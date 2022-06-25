"""Console script for tagupy."""
import os

import click

import tagupy
from tagupy.utils import is_project_dir


__all__ = [
    'generate',
]


@click.command(help='Generate Experience Matrix')
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
    if not is_project_dir(os.getcwd()):
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

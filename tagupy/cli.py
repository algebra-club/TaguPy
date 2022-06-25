import sys
from datetime import datetime

import click
from cookiecutter.main import cookiecutter

import tagupy
from tagupy.command import generate


@click.group()
def main():
    return 0


@main.command(help='Create New Project')
@click.option(
    '--name', '-n',
    help='New Project Name',
    prompt='project name?',
    required=True,
    type=click.STRING
)
@click.option(
    '--dryrun',
    default=False,
    help='Check results without process',
    type=click.BOOL
)
def new(name, dryrun):
    if dryrun:
        click.echo('project will be built with following variables')
        click.echo(f'prj_name: {name}, current_time: {datetime.now().strftime("%b-%d-%Y")}')
        return
    cookiecutter(
        f'{list(tagupy.__path__)[0]}/template',
        no_input=True,
        extra_context={
            "prj_name": name,
            "current_time": datetime.now().strftime('%b-%d-%Y'),
        }
    )


main.add_command(generate)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

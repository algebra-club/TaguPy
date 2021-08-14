"""Console script for tagupy."""
import sys
import click


@click.command()
def main(args=None):
    """Console script for tagupy."""
    click.echo("TODO: Manage your experiment workflow"
               "message sent from tagupy.cli.main")
    click.echo("See Also: docs at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

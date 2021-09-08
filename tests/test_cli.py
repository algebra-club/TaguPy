#!/usr/bin/env python

"""Tests for `tagupy` package."""

from click.testing import CliRunner
import pytest

from tagupy import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_cli(runner):
    '''
    Test cmd is callable
    '''
    result = runner.invoke(cli.main)
    assert result.exit_code == 0


def test_cli_help(runner):
    '''
    Test help option is callable
    '''
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


def test_cli_new(runner):
    '''
    Test new command with dryrun
    '''
    cmd = 'tagupy new --name nanachi --dryrun true'
    res = runner.invoke(cli.main, cmd.split(' ')[1:])
    assert res.exit_code == 0
    assert 'prj_name' in res.output
    assert 'nanachi' in res.output
    assert 'current_time' in res.output

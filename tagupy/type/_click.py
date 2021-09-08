import click
from tagupy.utils import is_positive_int, is_positive_int_list


class _PositiveIntList(click.types.ParamType):
    name = "positive_int_list"

    def convert(self, value, param, ctx):
        if is_positive_int_list(value):
            return value

        try:
            res = []
            for i in value.split(' '):
                res.append(int(i))
            if not is_positive_int_list(res):
                raise ValueError()
            return res
        except ValueError:
            self.fail(f"Invalid Input: {value!r} Each value should be positive integer", param, ctx)


class _PositiveInt(click.types.ParamType):
    name = "positive_int"

    def convert(self, value, param, ctx):
        if is_positive_int(value):
            return value

        try:
            res = int(value)
            if not is_positive_int(res):
                raise ValueError()
            return res
        except ValueError:
            self.fail(f"Invalid Input: {value!r} Value should be positive integer", param, ctx)


PositiveIntList = _PositiveIntList()
PositiveInt = _PositiveInt()

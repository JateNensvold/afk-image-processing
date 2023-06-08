from _pytest.terminal import TerminalReporter
from _pytest.config import Config
import random
import os
import typing
import pytest

from typing import TYPE_CHECKING


# if TYPE_CHECKING:
#     import pytest
#     import _pytest


def pytest_report_teststatus(report, config):
    messages = (
        'Egg and bacon',
        'Egg, sausage and bacon',
        'Egg and Spam',
        'Egg, bacon and Spam'
    )

    if report.when == 'teardown':
        line = f'{report.nodeid} says:\t"{random.choice(messages)}"'
        report.sections.append(('My custom section', line))


# class CustomTerminalReporter(TerminalReporter):
#     def short_test_summary(self) -> None:
#         self.write_sep("=", "my own short summary info")

#         failed = self.stats.get("failed", [])
#         for rep in failed:
#             self.write_line(f"failed test {rep.nodeid}")


def pytest_terminal_summary(terminalreporter: TerminalReporter, exitstatus: int, config: Config):
    # terminalreporter.getreports
    reports = terminalreporter.getreports('')
    # content = os.linesep.join(
    #     text for report in reports for secname, text in report.sections)
    # print([text for report in terminalreporter.getreports('failed') for secname, text in report.sections])
    print(terminalreporter.getreports('failed'))
    print([report.sections for report in terminalreporter.stats.get('failed')])
    # for report in reports:
    #     print(report.sections)
    # # print("reports", reports)

    # if content:
    #     terminalreporter.ensure_newline()
    #     terminalreporter.section(
    #         'My custom section', sep="-", blue=True, bold=True)
    #     terminalreporter.line(content)
    #     print(terminalreporter._tw.sep)
    pass
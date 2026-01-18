import textwrap

from runtime_introspect import runtime_feature_set


def pytest_report_header(config, start_path) -> list[str]:
    fs = runtime_feature_set()
    diagnostics = fs.diagnostics(features=["free-threading"])
    return [
        "Runtime optional features state (snapshot):",
        textwrap.indent("\n".join(diagnostics), "  "),
    ]

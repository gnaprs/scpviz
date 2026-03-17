"""Dash application entrypoint."""

from __future__ import annotations

from dash import Dash

from dash_app.layouts.main import build_layout


def create_app() -> Dash:
    app = Dash(__name__, title="scpviz Dash")
    app.layout = build_layout()

    # Import callback module after app layout has been declared.
    from dash_app.callbacks import main_callbacks  # noqa: F401

    return app


app = create_app()
server = app.server


if __name__ == "__main__":
    app.run(debug=True)


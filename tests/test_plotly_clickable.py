import numpy as np
import plotly.graph_objects as go

from drnb.plot.plotly import clickable_neighbors


def test_clickable_plotly_neighbors_creates_figure_widget() -> None:
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[0.0, 1.0, 2.0],
                y=[0.0, 1.0, 0.0],
                mode="markers",
                customdata=np.array([[0], [1], [2]]),
                marker={"size": 8, "symbol": "circle"},
            )
        ]
    )

    widget = clickable_neighbors(fig, np.array([[1], [0], [1]]))

    assert type(widget).__name__ == "FigureWidget"
    assert len(widget.data) == 1

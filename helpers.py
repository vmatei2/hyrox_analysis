import plotly.graph_objects as go

def create_distribution_figure(data_points, labels, xaxis_title, yaxis_title, title):
    fig = go.Figure()
    for i, data in enumerate(data_points):
        fig.add_trace(go.Scatter(
            x=[i] * len(data),
            y=data,
            mode='markers',
            name=f'{labels[i]}'
        ))
    fig.update_layout(
        xaxis=dict(
            tickvals=list(range(len(data_points))),
            ticktext=labels
        ),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        title=title,
        legend_title='Legend'
    )
    return fig

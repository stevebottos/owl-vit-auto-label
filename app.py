import glob

import plotly.express as px
from dash import Dash, dcc, html, Input, Output, no_update, ctx
import cv2

from detect import main


def start_app(query_impath, target_images_dir):
    app = Dash(__name__)

    img = cv2.imread(query_impath)[:, :, (2, 1, 0)]  # rgb
    fig = px.imshow(img)
    fig.update_layout(dragmode="drawrect")

    app.layout = html.Div(
        [
            dcc.Graph(id="graph-picture", figure=fig),
            html.Button("Run", id="btn-nclicks-1", n_clicks=0),
            html.Pre(id="annotations-data"),
            html.Div(id="container-button-timestamp"),
        ]
    )

    app.query_box = None

    def set_current_annotation(relayout_data):
        _box = relayout_data["shapes"].pop()
        app.query_box = [_box["x0"], _box["y0"], _box["x1"], _box["y1"]]
        print("Current bbox:", app.query_box)

    @app.callback(
        Output("annotations-data", "children"),
        Input("graph-picture", "relayoutData"),
        prevent_initial_call=True,
    )
    def on_new_annotation(relayout_data):
        if "shapes" in relayout_data:
            set_current_annotation(relayout_data)
        return no_update

    @app.callback(
        Output("container-button-timestamp", "children"),
        Input("btn-nclicks-1", "n_clicks"),
    )
    def displayClick(btn1):
        if "btn-nclicks-1" == ctx.triggered_id:
            print("Running inference")
            target_images = glob.glob(f"{target_images_dir}/*")
            main(query_impath, app.query_box, target_images)
        return

    app.run_server(debug=True)


if __name__ == "__main__":
    query_impath = "datasets/turosi/01GP25YV5DEHWSW0AJX6WTNRCR.jpeg"
    target_images_dir = "datasets/turosi"
    start_app(query_impath=query_impath, target_images_dir=target_images_dir)

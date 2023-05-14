import modules.scripts as scripts
import gradio as gr
import os

from modules import script_callbacks


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Row():
            angle = gr.Slider(
                minimum=0.0,
                maximum=360.0,
                step=1,
                value=0,
                label="Angle"
            )
            checkbox = gr.Checkbox(
                False,
                label="Checkbox"
            )
            btn = gr.Button(
                "Dummy image"
            ).style(
                full_width=False
            )
        with gr.Row():
            gallery = gr.Gallery(
                label="Dummy Image",
                show_label=False,
            )

        btn.click(
            dummy_images,
            inputs = None,
            outputs = [gallery],
        )

        return [(ui_component, "Extension Template", "extension_template_tab")]

def dummy_images():
    return [
        "https://chichi-pui.imgix.net/uploads/post_images/eee3b614-f126-4045-b53d-8bf38b98841d/05aba7f3-208b-4912-92f3-32d1bfc2edc3_1200x.jpeg?auto=format&lossless=0"
    ]


script_callbacks.on_ui_tabs(on_ui_tabs)

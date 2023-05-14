import modules.scripts as scripts
import gradio as gr
import os

from modules import script_callbacks


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Row():
            checkbox = gr.Checkbox(
                True,
                label="Show image"
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
            inputs = [checkbox],
            outputs = [gallery],
        )

        return [(ui_component, "Extension Example", "extension_example_tab")]

def dummy_images(checkbox):
    if (checkbox):
        return [
            "https://chichi-pui.imgix.net/uploads/post_images/eee3b614-f126-4045-b53d-8bf38b98841d/05aba7f3-208b-4912-92f3-32d1bfc2edc3_1200x.jpeg?auto=format&lossless=0"
        ]
    else:
        return []

script_callbacks.on_ui_tabs(on_ui_tabs)

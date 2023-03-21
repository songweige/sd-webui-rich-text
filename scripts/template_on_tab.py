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
            # TODO: add more UI components (cf. https://gradio.app/docs/#components)
        return [(ui_component, "Extension Template", "extension_template_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)

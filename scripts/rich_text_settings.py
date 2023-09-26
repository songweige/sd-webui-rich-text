import modules.scripts as scripts
import gradio as gr
import os

from modules import shared
from modules import script_callbacks

def on_ui_settings():
    section = ('template', "Rich-Text-to-Image")
    shared.opts.add_option(
        "option1",
        shared.OptionInfo(
            False,
            "This is a placeholder for option. It is not used yet.",
            gr.Checkbox,
            {"interactive": True},
            section=section)
    )

script_callbacks.on_ui_settings(on_ui_settings)

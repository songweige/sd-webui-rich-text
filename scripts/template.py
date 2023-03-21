import modules.scripts as scripts
import gradio as gr
import os

from modules import images, script_callbacks
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state

class ExtensionTemplateScript(scripts.Script):
        # Extension title in menu UI
        def title(self):
                return "Extension Template"

        # Decide to show menu in txt2img or img2img
        # - in "txt2img" -> is_img2img is `False`
        # - in "img2img" -> is_img2img is `True`
        #
        # below code always show extension menu
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def ui(self, is_img2img):
                with gr.Accordion('Extension Template', open=False):
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
                return [angle, checkbox]

        # Extension main process
        # Type: (StableDiffusionProcessing, List<UI>) -> (Processed)
        # args is [StableDiffusionProcessing, UI1, UI2, ...]
        def run(self, p, angle, checkbox):
                # TODO: get UI info through UI object angle, checkbox
                proc = process_images(p)
                # TODO: add image edit process via Processed object proc
                return proc


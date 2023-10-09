import modules.scripts as scripts
import gradio as gr
import os

from modules import script_callbacks
import math
import random
import os
import json
import time
import argparse
import torch
import numpy as np
from torchvision import transforms

from scripts.models.utils.attention_utils import get_token_maps
from scripts.models.region_diffusion import RegionDiffusion
from scripts.models.region_diffusion_xl import RegionDiffusionXL
from scripts.models.utils.richtext_utils import seed_everything, parse_json, get_region_diffusion_input,\
    get_attention_control_input, get_gradient_guidance_input


import gradio as gr
from PIL import Image, ImageOps
from share_btn import community_icon_html, loading_icon_html, share_js, css


help_text = """
If you are encountering an error or not achieving your desired outcome, here are some potential reasons and recommendations to consider:
1. If you format only a portion of a word rather than the complete word, an error may occur. 
2. If you use font color and get completely corrupted results, you may consider decrease the color weight lambda.
3. Consider using a different seed.
"""

canvas_html = """<iframe id='rich-text-root' style='width:100%' height='360px' src='file=./extensions/sd-webui-rich-text/rich-text-to-json-iframe.html' frameborder='0' scrolling='no'></iframe>"""
get_js_data = """
async (model_id, text_input, negative_prompt, num_segments, segment_threshold, inject_interval, inject_background, seed, color_guidance_weight, rich_text_input, height, width, steps, guidance_weights) => {
  const richEl = document.getElementById("rich-text-root");
  const data = richEl? richEl.contentDocument.body._data : {};
  return [model_id, text_input, negative_prompt, num_segments, segment_threshold, inject_interval, inject_background, seed, color_guidance_weight, JSON.stringify(data), height, width, steps, guidance_weights];
}
"""
set_js_data = """
async (text_input) => {
  const richEl = document.getElementById("rich-text-root");
  const data = text_input ? JSON.parse(text_input) : null;
  if (richEl && data) richEl.contentDocument.body.setQuillContents(data);
}
"""

get_window_url_params = """
async (url_params) => {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    return [url_params];
}
"""

class RichText2Img():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = RegionDiffusion(self.device, 'runwayml/stable-diffusion-v1-5')
        self.model = None


def load_url_params(url_params):
    if 'prompt' in url_params:
        return gr.update(visible=True), url_params
    else:
        return gr.update(visible=False), url_params


def on_ui_tabs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = RegionDiffusion(device, 'runwayml/stable-diffusion-v1-5')
    # model = RegionDiffusionXL()
    richtext2img = RichText2Img()

    def generate(
        model_id: str,
        text_input: str,
        negative_text: str,
        num_segments: int,
        segment_threshold: float,
        inject_interval: float,
        inject_background: float,
        seed: int,
        color_guidance_weight: float,
        rich_text_input: str,
        height: int,
        width: int,
        steps: int,
        guidance_weight: float,
    ):
        model = richtext2img.model
        if model_id == "runwayml/stable-diffusion-v1-5":
            if model is None or model.model_id != model_id:
                richtext2img.model = RegionDiffusion(device, model_id)
                model = richtext2img.model
            if width == 0 or height == 0:
                width = 512
                height = 512
        elif model_id in ["stabilityai/stable-diffusion-xl-base-1.0", "Linaqruf/animagine-xl"]:
            if  model is None or model.model_id != model_id:
                richtext2img.model = RegionDiffusionXL(model_id)
                model = richtext2img.model
            if width == 0 or height == 0:
                width = 1024
                height = 1024
        else:
            raise gr.Error("Please select a model.")
        run_dir = 'results/'
        os.makedirs(run_dir, exist_ok=True)
        steps = 41 if not steps else steps
        guidance_weight = 8.5 if not guidance_weight else guidance_weight
        text_input = rich_text_input if rich_text_input != '' and rich_text_input != None else text_input
        print('text_input', text_input, width, height, steps, guidance_weight, num_segments, segment_threshold, inject_interval, inject_background, color_guidance_weight, negative_text)
        if (text_input == '' or rich_text_input == ''):
            raise gr.Error("Please enter some text.")
        # parse json to span attributes
        base_text_prompt, style_text_prompts, footnote_text_prompts, footnote_target_tokens,\
            color_text_prompts, color_names, color_rgbs, size_text_prompts_and_sizes, use_grad_guidance = parse_json(
                json.loads(text_input))

        # create control input for region diffusion
        region_text_prompts, region_target_token_ids, base_tokens = get_region_diffusion_input(
            model, base_text_prompt, style_text_prompts, footnote_text_prompts,
            footnote_target_tokens, color_text_prompts, color_names)

        # create control input for cross attention
        text_format_dict = get_attention_control_input(
            model, base_tokens, size_text_prompts_and_sizes)

        # create control input for region guidance
        text_format_dict, color_target_token_ids = get_gradient_guidance_input(
            model, base_tokens, color_text_prompts, color_rgbs, text_format_dict, color_guidance_weight=color_guidance_weight)

        seed_everything(seed)

        # get token maps from plain text to image generation.
        begin_time = time.time()
        if model.selfattn_maps is None and model.crossattn_maps is None:
            model.remove_tokenmap_hooks()
            model.register_tokenmap_hooks()
        else:
            model.remove_tokenmap_hooks()
            model.remove_tokenmap_hooks()
        if model_id == "runwayml/stable-diffusion-v1-5":
            plain_img = model.produce_attn_maps([base_text_prompt], [negative_text],
                                                height=height, width=width, num_inference_steps=steps,
                                                guidance_scale=guidance_weight)
        else:
            plain_img = model.sample([base_text_prompt], negative_prompt=[negative_text],
                                    height=height, width=width, num_inference_steps=steps,
                                    guidance_scale=guidance_weight, run_rich_text=False)
        print('time lapses to get attention maps: %.4f' %
              (time.time()-begin_time))
        seed_everything(seed)
        color_obj_masks, segments_vis, token_maps = get_token_maps(model.selfattn_maps, model.crossattn_maps, model.n_maps, run_dir,
                                                                   height//8, width//8, color_target_token_ids[:-1], seed,
                                                                   base_tokens, segment_threshold=segment_threshold, num_segments=num_segments,
                                                                   return_vis=True)
        seed_everything(seed)
        model.masks, segments_vis, token_maps = get_token_maps(model.selfattn_maps, model.crossattn_maps, model.n_maps, run_dir,
                                                               height//8, width//8, region_target_token_ids[:-1], seed,
                                                               base_tokens, segment_threshold=segment_threshold, num_segments=num_segments,
                                                               return_vis=True)
        color_obj_atten_all = torch.zeros_like(color_obj_masks[-1])
        for obj_mask in color_obj_masks[:-1]:
            color_obj_atten_all += obj_mask
        color_obj_masks = [transforms.functional.resize(color_obj_mask, (height, width),
                                                        interpolation=transforms.InterpolationMode.BICUBIC,
                                                        antialias=True)
                           for color_obj_mask in color_obj_masks]
        text_format_dict['color_obj_atten'] = color_obj_masks
        text_format_dict['color_obj_atten_all'] = color_obj_atten_all
        model.remove_tokenmap_hooks()

        # generate image from rich text
        begin_time = time.time()
        seed_everything(seed)
        if model_id == "runwayml/stable-diffusion-v1-5":
            rich_img = model.prompt_to_img(region_text_prompts, [negative_text],
                                        height=height, width=width, num_inference_steps=steps,
                                        guidance_scale=guidance_weight, use_guidance=use_grad_guidance,
                                        inject_selfattn=inject_interval, text_format_dict=text_format_dict,
                                        inject_background=inject_background)
            print('time lapses to generate image from rich text: %.4f' %
                (time.time()-begin_time))
            return [plain_img[0], rich_img[0], segments_vis, token_maps]
        else:
            rich_img = model.sample(region_text_prompts, negative_prompt=[negative_text],
                                        height=height, width=width, num_inference_steps=steps,
                                        guidance_scale=guidance_weight, use_guidance=use_grad_guidance,
                                        inject_selfattn=inject_interval, text_format_dict=text_format_dict,
                                        inject_background=inject_background, run_rich_text=True)
            print('time lapses to generate image from rich text: %.4f' %
                (time.time()-begin_time))
            return [plain_img.images[0], rich_img.images[0], segments_vis, token_maps]
    
    with gr.Blocks(analytics_enabled=False) as ui_component:
        url_params = gr.JSON({}, visible=False, label="URL Params")
        gr.HTML("""<h1 style="font-weight: 900; margin-bottom: 7px;">Expressive Text-to-Image Generation with Rich Text</h1>
                   <p> <a href="https://rich-text-to-image.github.io">[Website]</a> | <a href="https://github.com/SongweiGe/rich-text-to-image">[Code]</a> | <a href="https://arxiv.org/abs/2304.06720">[Paper]</a><p/>
                """)
        with gr.Row():
            with gr.Column():
                rich_text_el = gr.HTML(canvas_html, elem_id="canvas_html")
                rich_text_input = gr.Textbox(value="", visible=False)
                text_input = gr.Textbox(
                    label='Rich-text JSON Input',
                    visible=False,
                    max_lines=1,
                    placeholder='Example: \'{"ops":[{"insert":"a Gothic "},{"attributes":{"color":"#b26b00"},"insert":"church"},{"insert":" in a the sunset with a beautiful landscape in the background.\n"}]}\'',
                    elem_id="text_input"
                )
                model_id = gr.Radio(choices=["runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-xl-base-1.0", "Linaqruf/animagine-xl"], value="runwayml/stable-diffusion-v1-5", label="Model ID", elem_id="model_id")
                negative_prompt = gr.Textbox(
                    label='Negative Prompt',
                    max_lines=1,
                    placeholder='Example: poor quality, blurry, dark, low resolution, low quality, worst quality',
                    elem_id="negative_prompt"
                )
                segment_threshold = gr.Slider(label='Token map threshold',
                                              info='(See less area in token maps? Decrease this. See too much area? Increase this.)',
                                              minimum=0,
                                              maximum=1,
                                              step=0.01,
                                              value=0.45)
                inject_interval = gr.Slider(label='Detail preservation',
                                            info='(To preserve more structure from plain-text generation, increase this. To see more rich-text attributes, decrease this.)',
                                            minimum=0,
                                            maximum=1,
                                            step=0.01,
                                            value=0.3)
                inject_background = gr.Slider(label='Unformatted token preservation',
                                            info='(To affect less the tokens without any rich-text attributes, increase this.)',
                                            minimum=0,
                                            maximum=1,
                                            step=0.01,
                                            value=0.3)
                color_guidance_weight = gr.Slider(label='Color weight',
                                                  info='(To obtain more precise color, increase this, while too large value may cause artifacts.)',
                                                  minimum=0,
                                                  maximum=2,
                                                  step=0.1,
                                                  value=0.5)
                num_segments = gr.Slider(label='Number of segments',
                                         minimum=2,
                                         maximum=20,
                                         step=1,
                                         value=9)
                seed = gr.Slider(label='Seed',
                                 minimum=0,
                                 maximum=100000,
                                 step=1,
                                 value=6,
                                 elem_id="seed"
                                 )
                with gr.Accordion('Other Parameters', open=False):
                    steps = gr.Slider(label='Number of Steps',
                                      minimum=0,
                                      maximum=500,
                                      step=1,
                                      value=41)
                    guidance_weight = gr.Slider(label='CFG weight',
                                                minimum=0,
                                                maximum=50,
                                                step=0.1,
                                                value=8.5)
                    width = gr.Slider(minimum=64, maximum=2048, step=8, label="Width", value=0)
                    height = gr.Slider(minimum=64, maximum=2048, step=8, label="Height", value=0)

                with gr.Row():
                    with gr.Column(scale=1, min_width=100):
                        generate_button = gr.Button("Generate")
            with gr.Column():
                richtext_result = gr.Image(
                    label='Rich-text', elem_id="rich-text-image")
                richtext_result.style(height=784)
                with gr.Row():
                    plaintext_result = gr.Image(
                        label='Plain-text', elem_id="plain-text-image")
                    segments = gr.Image(label='Segmentation')
                with gr.Row():
                    token_map = gr.Image(label='Token Maps')

        generate_button.click(fn=lambda: gr.update(visible=False), inputs=None, queue=False).then(
            fn=generate,
            inputs=[
                model_id,
                text_input,
                negative_prompt,
                num_segments,
                segment_threshold,
                inject_interval,
                inject_background,
                seed,
                color_guidance_weight,
                rich_text_input,
                height,
                width,
                steps,
                guidance_weight,
            ],
            outputs=[plaintext_result, richtext_result, segments, token_map],
            _js=get_js_data
        ).then(
            fn=lambda: gr.update(visible=True), inputs=None, queue=False)
        text_input.change(
            fn=None, inputs=[text_input], outputs=None, _js=set_js_data, queue=False)
        return [(ui_component, "rich-text2img", "rich_text2img")]

script_callbacks.on_ui_tabs(on_ui_tabs)

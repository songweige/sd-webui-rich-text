def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_components:
        return [(ui_components, "Extension Template", "extension_template_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)

# stable-diffusion-webui-extension-templates

a template of stable-diffusion-webui extension for understand and develop  quickly 

## basic design
```
├── install.py (optional)
└── scripts
    ├── ${extension_name}.py
    ... (if extension need module division)
```

## Pattern 1. custom script

<img src="https://user-images.githubusercontent.com/128375799/226570836-8a9c5640-5258-4b4e-9cbe-e139732d8419.png"  width="600"/>

see `scripts/template.py`

## Pattern 2. ui on tab

<img src="https://user-images.githubusercontent.com/128375799/226570948-578706a3-a278-4228-a999-6147050f5706.png"  width="600"/>

see `scripts/template_on_tab.py`


## Pattern 3. ui on settings

<img src="https://user-images.githubusercontent.com/128375799/235482423-61c15a3e-8239-41f8-bffb-16efaafc084d.png"  width="600"/>
<img src="https://user-images.githubusercontent.com/128375799/235482466-9f7822f3-1ea9-4d1e-bbf5-cb98de041119.png"  width="600"/>


see `scripts/template_on_settings.py`

## Ref.
- [Developing custom scripts · AUTOMATIC1111/stable-diffusion-webui Wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Developing-custom-scripts)
- [Developing extensions · AUTOMATIC1111/stable-diffusion-webui Wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Developing-extensions)
- [Extensions · AUTOMATIC1111/stable-diffusion-webui Wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Extensions) (to read real extension code)

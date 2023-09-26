import launch

# TODO: add pip dependency if need extra module only on extension

if not launch.is_installed("diffusers"):
    launch.run_pip("install diffusers==0.18.2", "requirements for Rich-Text-to-Image")

if not launch.is_installed("invisible-watermark"):
    launch.run_pip("install invisible-watermark==0.2.0", "requirements for Rich-Text-to-Image")

if not launch.is_installed("accelerate"):
    launch.run_pip("install accelerate==0.21.0", "requirements for Rich-Text-to-Image")

if not launch.is_installed("safetensors"):
    launch.run_pip("install safetensors==0.3.1", "requirements for Rich-Text-to-Image")

if not launch.is_installed("seaborn"):
    launch.run_pip("install seaborn==0.12.2", "requirements for Rich-Text-to-Image")

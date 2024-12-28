# Heurist Image

A Python implementation of the Heurist Image API containing two main classes: 
- `ImageGen` for basic image generation.
- `SmartGen` for enhanced image generation with additional parameters.

## Installation
Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Prerequisites

- Python 3.7+
- Heurist API key
- Environment variables set up in a `.env` file

## Environment Setup

Create a `.env` file in your project root with the following variables:

```bash
HEURIST_API_KEY=your_api_key
HEURIST_SEQUENCER_URL=https://api.heurist.ai/sequencer
```

## Usage

### Basic Image Generation

```python
import asyncio
from heurist_image import ImageGen
async def generate_basic_image():
async with ImageGen(api_key="your_api_key") as generator:
response = await generator.generate({
    "model": "FLUX.1-dev",
    "prompt": "A serene Japanese garden with cherry blossoms",
    "width": 1024,
    "height": 768,
    "num_iterations": 20,
    "guidance_scale": 7.5
})
print(f"Generated image URL: {response['url']}")
asyncio.run(generate_basic_image())
```

### Enhanced Image Generation (SmartGen)
```python
import asyncio
from heurist_image import SmartGen
async def generate_smart_image():
async with SmartGen(api_key="your_api_key") as generator:
response = await generator.generate_image(
    description="A futuristic cyberpunk cityscape",
    image_model="FLUX.1-dev",
    stylization_level=4,
    detail_level=5,
    color_level=5,
    lighting_level=4,
    must_include="neon lights, flying cars",
    quality="high"
)
print(f"Generated image URL: {response['url']}")
asyncio.run(generate_smart_image())
```

## Testing

Run the test suite using:

```bash
python -m pytest heurist_image/test_image_gen.py
```
## API Reference

### ImageGen Class

Basic image generation with customizable parameters:
- `prompt`: Text description of the desired image
- `model`: AI model to use (e.g., "FLUX.1-dev")
- `width`: Image width in pixels
- `height`: Image height in pixels
- `num_iterations`: Number of iterations for generation
- `guidance_scale`: Controls how closely the image follows the prompt
- `neg_prompt`: Negative prompt (optional)
- `seed`: Random seed for reproducibility (optional)

### SmartGen Class

Enhanced image generation with additional controls:
- `description`: Text description of the desired image
- `image_model`: AI model to use
- `width`: Image width in pixels (default: 1024)
- `height`: Image height in pixels (default: 768)
- `stylization_level`: Control over artistic style (optional)
- `detail_level`: Control over image detail (optional)
- `color_level`: Control over color intensity (optional)
- `lighting_level`: Control over lighting effects (optional)
- `must_include`: Required elements in the image (optional)
- `quality`: Image quality setting (default: "normal")

## For more info and parameters see the Heurist API documentation:
https://docs.heurist.ai/dev-guide/heurist-sdk/smartgen

## License

This project is licensed under the MIT License - see the LICENSE file for details.
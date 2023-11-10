import torch
from flask import Flask, render_template, request
from diffusers import StableDiffusionPipeline
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.manual_seed(seed)
    image_gen_steps = 50  
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 15

image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float32,  # Changed torch_dtype to float32
    revision="fp16", use_auth_token='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', guidance_scale=CFG.image_gen_guidance_scale
)
image_gen_model = image_gen_model.to(CFG.device)

def generate_image(prompt, model, num_steps=CFG.image_gen_steps, guidance_scale=CFG.image_gen_guidance_scale):
    image = model(
        prompt, num_inference_steps=num_steps,
        generator=CFG.generator,
        guidance_scale=guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image

def generate_images_parallel(prompts, model):
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(generate_image, prompts, [model] * len(prompts)))
    return images

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_image', methods=['POST'])
def generate_image_route():
    prompt = request.form['prompt']
    num_steps = int(request.form['num_steps'])
    guidance_scale = int(request.form['guidance_scale'])

    
    prompts = [prompt] * 5  
    enhanced_images = generate_images_parallel(prompts, image_gen_model)


    image_datas = []
    for enhanced_image in enhanced_images:
        image_io = io.BytesIO()
        enhanced_image.save(image_io, format='PNG')
        image_data = base64.b64encode(image_io.getvalue()).decode('utf-8')
        image_datas.append(image_data)

    return render_template('index.html', image_datas=image_datas)

if __name__ == '__main__':
    app.run(debug=False)

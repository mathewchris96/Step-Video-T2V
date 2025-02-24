import os
import threading
import time
import argparse
import pickle

from flask import Flask, Response, request, Blueprint
from flask_restful import Api, Resource

import torch

# Set device and dtype
device = f'cuda:{torch.cuda.device_count()-1}'
torch.cuda.set_device(device)
dtype = torch.bfloat16

def parsed_args():
    parser = argparse.ArgumentParser(description="StepVideo API Functions")
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--clip_dir', type=str, default='hunyuan_clip')
    parser.add_argument('--llm_dir', type=str, default='step_llm')
    parser.add_argument('--vae_dir', type=str, default='vae')
    parser.add_argument('--port', type=str, default='8080')
    args = parser.parse_args()
    return args

##############################################
# Modified Pipeline Classes with Lazy Loading
##############################################

class StepVaePipeline(Resource):
    def __init__(self, vae_dir, version=2):
        # Initially, do not load the model
        self.vae_dir = vae_dir
        self.version = version
        self.vae = None
        self.scale_factor = 1.0
        self.loaded = False

    def load_model(self):
        from stepvideo.vae.vae import AutoencoderKL
        (model_name, z_channels) = ("vae_v2.safetensors", 64) if self.version == 2 else ("vae.safetensors", 16)
        model_path = os.path.join(self.vae_dir, model_name)
        self.vae = AutoencoderKL(
            z_channels=z_channels,
            model_path=model_path,
            version=self.version,
        ).to(dtype).to(device).eval()
        print("Initialized VAE model.")
        self.loaded = True

    def decode(self, samples, *args, **kwargs):
        if not self.loaded or self.vae is None:
            return "VAE model is still loading."
        with torch.no_grad():
            try:
                model_dtype = next(self.vae.parameters()).dtype
                model_device = next(self.vae.parameters()).device
                samples = self.vae.decode(samples.to(model_dtype).to(model_device) / self.scale_factor)
                if hasattr(samples, 'sample'):
                    samples = samples.sample
                return samples
            except Exception as e:
                print("Error during VAE decoding:", e)
                torch.cuda.empty_cache()
                return None

lock = threading.Lock()

class VAEapi(Resource):
    def __init__(self, vae_pipeline):
        self.vae_pipeline = vae_pipeline

    def get(self):
        with lock:
            try:
                feature = pickle.loads(request.get_data())
                feature['api'] = 'vae'
                feature = {k: v for k, v in feature.items() if v is not None}
                video_latents = self.vae_pipeline.decode(**feature)
                if isinstance(video_latents, str):  # Model still loading
                    return Response(video_latents, status=503)
                response = pickle.dumps(video_latents)
            except Exception as e:
                print("Caught Exception in VAEapi:", e)
                return Response(str(e), status=500)
            return Response(response)

class CaptionPipeline(Resource):
    def __init__(self, llm_dir, clip_dir):
        # Initially, do not load heavy models.
        self.llm_dir = llm_dir
        self.clip_dir = clip_dir
        self.text_encoder = None
        self.clip = None
        self.loaded = False

    def load_models(self):
        from stepvideo.text_encoder.stepllm import STEP1TextEncoder
        from stepvideo.text_encoder.clip import HunyuanClip
        self.text_encoder = STEP1TextEncoder(self.llm_dir, max_length=320).to(dtype).to(device).eval()
        print("Initialized text encoder.")
        self.clip = HunyuanClip(self.clip_dir, max_length=77).to(device).eval()
        print("Initialized clip encoder.")
        self.loaded = True

    def embedding(self, prompts, *args, **kwargs):
        if not self.loaded or self.text_encoder is None or self.clip is None:
            return "Caption model is still loading."
        with torch.no_grad():
            try:
                y, y_mask = self.text_encoder(prompts)
                clip_embedding, _ = self.clip(prompts)
                len_clip = clip_embedding.shape[1]
                y_mask = torch.nn.functional.pad(y_mask, (len_clip, 0), value=1)  # pad attention_mask with clip's length 
                data = {
                    'y': y.detach().cpu(),
                    'y_mask': y_mask.detach().cpu(),
                    'clip_embedding': clip_embedding.to(torch.bfloat16).detach().cpu()
                }
                return data
            except Exception as err:
                print("Error during caption embedding:", err)
                return None

lock = threading.Lock()

class Captionapi(Resource):
    def __init__(self, caption_pipeline):
        self.caption_pipeline = caption_pipeline

    def get(self):
        with lock:
            try:
                feature = pickle.loads(request.get_data())
                feature['api'] = 'caption'
                feature = {k: v for k, v in feature.items() if v is not None}
                embeddings = self.caption_pipeline.embedding(**feature)
                if isinstance(embeddings, str):  # Model still loading
                    return Response(embeddings, status=503)
                response = pickle.dumps(embeddings)
            except Exception as e:
                print("Caught Exception in Captionapi:", e)
                return Response(str(e), status=500)
            return Response(response)

##############################################
# Remote Server Definition
##############################################

class RemoteServer(object):
    def __init__(self, args) -> None:
        self.app = Flask(__name__)
        root = Blueprint("root", __name__)
        self.app.register_blueprint(root)
        api = Api(self.app)
        
        self.vae_pipeline = StepVaePipeline(
            vae_dir=os.path.join(args.model_dir, args.vae_dir)
        )
        api.add_resource(
            VAEapi,
            "/vae-api",
            resource_class_args=[self.vae_pipeline],
        )
        
        self.caption_pipeline = CaptionPipeline(
            llm_dir=os.path.join(args.model_dir, args.llm_dir),
            clip_dir=os.path.join(args.model_dir, args.clip_dir)
        )
        api.add_resource(
            Captionapi,
            "/caption-api",
            resource_class_args=[self.caption_pipeline],
        )

    def run(self, host="0.0.0.0", port=8080):
        self.app.run(host, port=port, threaded=True, debug=False)

# Function to start heavy model loading in background threads.
def start_model_loading(remote_server_instance):
    # Start VAE loading in one thread:
    threading.Thread(target=remote_server_instance.vae_pipeline.load_model, daemon=True).start()
    # Start Caption models loading in another thread:
    threading.Thread(target=remote_server_instance.caption_pipeline.load_models, daemon=True).start()

if __name__ == "__main__":
    args = parsed_args()
    flask_server = RemoteServer(args)
    
    # Start heavy model loading in background.
    start_model_loading(flask_server)
    
    print("🔥 Starting Flask server on 0.0.0.0:8080 with endpoints:")
    for rule in flask_server.app.url_map.iter_rules():
        print("➡️", rule)
    flask_server.run(host="0.0.0.0", port=args.port)

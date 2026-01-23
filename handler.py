import sys
import traceback
import os

# --- CRASH PROOFING ---
# We wrap imports so the container starts even if libraries fail.
INIT_ERROR = None

try:
    import base64
    import torch
    import runpod
    import requests
    import soundfile as sf
    import numpy as np
    from io import BytesIO
    from PIL import Image

    # Pipeline Imports
    from diffusers import DiffusionPipeline
    from diffusers.utils import export_to_video
    
    # MOVIEPY FIX: We rely on moviepy==1.0.3 so this import works again
    from moviepy.editor import VideoFileClip, AudioFileClip

except Exception as e:
    # Capture the error but DO NOT CRASH. 
    # We will return this error to the user when they send a request.
    INIT_ERROR = traceback.format_exc()
    print(f"CRITICAL IMPORT ERROR: {INIT_ERROR}", flush=True)

# --- CONFIGURATION ---
VOLUME_CACHE = "/runpod-volume/huggingface-cache"
OUTPUT_DIR = "/runpod-volume/outputs"

# --- CRITICAL FIX: CHANGED MODEL ID ---
# "Lightricks/LTX-2" is currently broken/incomplete on HF.
# "Lightricks/LTX-Video" is the correct, stable repository.
MODEL_ID = "Lightricks/LTX-Video"

# Global Cache
pipe = None

def load_pipeline():
    global pipe
    if pipe is None:
        print(f"Loading LTX-Video Model ({MODEL_ID})...", flush=True)
        try:
            # We use from_pretrained with bfloat16 for efficiency
            pipe = DiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                cache_dir=VOLUME_CACHE,
                # variant="fp8" # Optional: Use this if you want the FP8 quantized version for speed
            ).to("cuda")
            print("--- LTX-VIDEO LOADED SUCCESSFULLY ---", flush=True)
        except Exception as e:
            print(f"FAILED to load model: {e}")
            raise e

def smart_resize(img):
    """Resizes image to 1024px-long edge, ensuring dimensions are multiples of 32."""
    width, height = img.size
    aspect_ratio = width / height
    target_long_edge = 1024
    
    if width > height:
        new_width = target_long_edge
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_long_edge
        new_width = int(new_height * aspect_ratio)
    
    new_width = new_width - (new_width % 32)
    new_height = new_height - (new_height % 32)
    return img.resize((new_width, new_height), Image.LANCZOS), new_width, new_height

def save_audiovisual_output(video_frames, audio_data, audio_sample_rate, output_path):
    temp_video_path = output_path.replace(".mp4", "_temp_vid.mp4")
    export_to_video(video_frames, temp_video_path, fps=24)
    
    temp_audio_path = output_path.replace(".mp4", "_temp_audio.wav")
    
    # Ensure audio is CPU numpy array
    if torch.is_tensor(audio_data):
        audio_data = audio_data.cpu().numpy()
        
    if len(audio_data.shape) > 1 and audio_data.shape[0] == 1:
        audio_data = audio_data.squeeze(0)
        
    sf.write(temp_audio_path, audio_data, audio_sample_rate)
    
    try:
        video_clip = VideoFileClip(temp_video_path)
        audio_clip = AudioFileClip(temp_audio_path)
        
        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.subclip(0, video_clip.duration)
            
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        
        video_clip.close()
        audio_clip.close()
        final_clip.close()
        
        os.remove(temp_video_path)
        os.remove(temp_audio_path)
        return True
    except Exception as e:
        print(f"Merge Error: {e}")
        return False

def handler(job):
    print(f"--- JOB START: {job['id']} ---", flush=True)
    
    # 1. FAIL FAST CHECK
    if INIT_ERROR:
        return {"error": "Container failed to initialize imports", "traceback": INIT_ERROR}

    try:
        load_pipeline()
        
        job_input = job["input"]
        prompt = job_input.get("prompt", "A cinematic scene")
        negative_prompt = job_input.get("negative_prompt", "low quality, worst quality, deformed")
        image_url = job_input.get("image_url", None)
        image_b64 = job_input.get("image_base64", None)
        num_frames = job_input.get("num_frames", 121)
        guidance_scale = float(job_input.get("guidance_scale", 3.0))
        
        output_path = f"{OUTPUT_DIR}/{job['id']}.mp4"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        input_image = None
        if image_url:
            input_image = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")
        elif image_b64:
            input_image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")

        with torch.no_grad():
            if input_image:
                print(f"Running Image-to-Video generation...", flush=True)
                processed_img, w, h = smart_resize(input_image)
                output = pipe(
                    image=processed_img,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    height=h,
                    width=w,
                    guidance_scale=guidance_scale,
                    num_inference_steps=40
                )
            else:
                print(f"Running Text-to-Video generation...", flush=True)
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    height=720,
                    width=1280,
                    guidance_scale=guidance_scale,
                    num_inference_steps=40
                )

        video_frames = output.frames[0]
        
        # Check for audio output (LTX-Video currently is video-only, but keeping this logic for future)
        if hasattr(output, 'audios') and output.audios is not None:
            audio_data = output.audios[0] 
            sample_rate = getattr(output, 'audio_sampling_rate', 16000)
            save_audiovisual_output(video_frames, audio_data, sample_rate, output_path)
        else:
            # Silent video export
            export_to_video(video_frames, output_path, fps=24)

        with open(output_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")
            
        return {"status": "success", "video_base64": video_b64}

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    if 'runpod' in sys.modules:
        runpod.serverless.start({"handler": handler})
    else:
        print("CRITICAL: RunPod library not found.")
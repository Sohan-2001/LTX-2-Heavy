import sys
import traceback
import os
import scipy.io.wavfile as wav
import sys
import traceback
import os
import scipy.io.wavfile as wav

# --- CRASH PROOFING & IMPORTS ---
INIT_ERROR = None
try:
    import base64
    import torch
    import runpod
    import requests
    import numpy as np
    from io import BytesIO
    from PIL import Image
    from diffusers import LTXVideoPipeline, LTXImageToVideoPipeline
    from diffusers.utils import export_to_video
    from moviepy.editor import VideoFileClip, AudioFileClip

except Exception as e:
    INIT_ERROR = traceback.format_exc()
    print(f"CRITICAL IMPORT ERROR: {INIT_ERROR}", flush=True)

# --- CONFIGURATION ---
VOLUME_CACHE = "/runpod-volume/huggingface-cache"
OUTPUT_DIR = "/runpod-volume/outputs"
MODEL_ID = "Lightricks/LTX-Video"

pipe_t2v = None
pipe_i2v = None

def load_pipelines():
    global pipe_t2v, pipe_i2v
    if pipe_t2v is None:
        print(f"Loading LTX-Video Pipelines ({MODEL_ID})...", flush=True)
        try:
            # 1. Text-to-Video Pipeline
            pipe_t2v = LTXVideoPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                cache_dir=VOLUME_CACHE,
                variant="fp16" # Use fp16 variant if available for speed/quality balance
            ).to("cuda")

            # 2. Image-to-Video Pipeline (shares components to save RAM)
            pipe_i2v = LTXImageToVideoPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                cache_dir=VOLUME_CACHE,
                variant="fp16",
                transformer=pipe_t2v.transformer,
                vae=pipe_t2v.vae,
                scheduler=pipe_t2v.scheduler,
                text_encoder=pipe_t2v.text_encoder,
                tokenizer=pipe_t2v.tokenizer
            ).to("cuda")
            
            print("--- LTX-VIDEO PIPELINES LOADED ---", flush=True)
        except Exception as e:
            print(f"FAILED to load model: {e}")
            raise e

def save_audiovisual_output(video_path, audio_tensor, sample_rate, output_path):
    """Merges video and audio into a final MP4."""
    try:
        # 1. Save Audio to temporary WAV
        temp_audio_path = output_path.replace(".mp4", "_audio.wav")
        
        # Normalize audio if needed
        audio_data = audio_tensor.cpu().numpy().squeeze()
        # Scale to 16-bit integer range if it's float [-1, 1]
        if audio_data.dtype == np.float32:
             audio_data = (audio_data * 32767).astype(np.int16)
             
        wav.write(temp_audio_path, sample_rate, audio_data)
        
        # 2. Merge using MoviePy
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(temp_audio_path)
        
        # Trim audio to match video length
        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.subclip(0, video_clip.duration)
            
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        
        # Cleanup
        video_clip.close()
        audio_clip.close()
        final_clip.close()
        os.remove(temp_audio_path)
        os.remove(video_path) # Remove silent video source
        
        return True
    except Exception as e:
        print(f"Audio Merge Failed: {e}")
        # If merge fails, return the silent video at least
        if os.path.exists(video_path):
            os.rename(video_path, output_path)
        return False

def handler(job):
    print(f"--- JOB START: {job['id']} ---", flush=True)
    if INIT_ERROR: return {"error": "Init failed", "traceback": INIT_ERROR}

    try:
        load_pipelines()
        
        job_input = job["input"]
        prompt = job_input.get("prompt", "A cinematic scene")
        negative_prompt = job_input.get("negative_prompt", "low quality, blurry, deformed, watermark")
        image_url = job_input.get("image_url", None)
        image_b64 = job_input.get("image_base64", None)
        
        # QUALITY SETTINGS
        # Increased steps for better quality (default was 40)
        num_inference_steps = int(job_input.get("num_inference_steps", 50)) 
        # Slightly higher guidance for sharper adherence
        guidance_scale = float(job_input.get("guidance_scale", 3.5))
        
        output_path = f"{OUTPUT_DIR}/{job['id']}.mp4"
        temp_video_path = f"{OUTPUT_DIR}/{job['id']}_silent.mp4"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Prepare Input Image
        input_image = None
        if image_url:
            input_image = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")
        elif image_b64:
            input_image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")

        # GENERATION
        with torch.no_grad():
            if input_image:
                print("Running Image-to-Video...", flush=True)
                # Resize image to be divisible by 32
                w, h = input_image.size
                w = w - (w % 32)
                h = h - (h % 32)
                input_image = input_image.resize((w, h))
                
                output = pipe_i2v(
                    image=input_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    output_type="pt" # Return tensors to handle audio manually if needed
                )
            else:
                print("Running Text-to-Video...", flush=True)
                # Standard resolution for LTX (768x512 is most stable)
                output = pipe_t2v(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=512, 
                    width=768,
                    num_frames=161, # ~6 seconds
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )

        # SAVE VIDEO
        # Access the video frames (it returns a VideoPipelineOutput)
        video_frames = output.frames[0]
        export_to_video(video_frames, temp_video_path, fps=24)
        
        # HANDLE AUDIO
        # LTX-Video generates audio if the prompt implies it, usually returned in .audios
        # Note: Current public LTX-Video pipeline in diffusers MIGHT NOT expose audio output directly 
        # depending on the version installed. 
        # If 'audios' attribute exists, we process it.
        
        has_audio = False
        if hasattr(output, "audios") and output.audios is not None:
             print("Audio generated! Merging...", flush=True)
             # Assumption: output.audios is a list of tensors/arrays
             audio_tensor = output.audios[0]
             # Default LTX sample rate is usually 16kHz or 24kHz
             sample_rate = getattr(output, "audio_sampling_rate", 16000)
             
             if save_audiovisual_output(temp_video_path, audio_tensor, sample_rate, output_path):
                 has_audio = True
        
        if not has_audio:
            print("No audio generated or audio merge skipped.", flush=True)
            if os.path.exists(temp_video_path):
                os.rename(temp_video_path, output_path)

        # Return Base64
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
# --- CRASH PROOFING & IMPORTS ---
INIT_ERROR = None
try:
    import base64
    import torch
    import runpod
    import requests
    import numpy as np
    from io import BytesIO
    from PIL import Image
    from diffusers import LTXVideoPipeline, LTXImageToVideoPipeline
    from diffusers.utils import export_to_video
    from moviepy.editor import VideoFileClip, AudioFileClip

except Exception as e:
    INIT_ERROR = traceback.format_exc()
    print(f"CRITICAL IMPORT ERROR: {INIT_ERROR}", flush=True)

# --- CONFIGURATION ---
VOLUME_CACHE = "/runpod-volume/huggingface-cache"
OUTPUT_DIR = "/runpod-volume/outputs"
MODEL_ID = "Lightricks/LTX-Video"

pipe_t2v = None
pipe_i2v = None

def load_pipelines():
    global pipe_t2v, pipe_i2v
    if pipe_t2v is None:
        print(f"Loading LTX-Video Pipelines ({MODEL_ID})...", flush=True)
        try:
            # 1. Text-to-Video Pipeline
            pipe_t2v = LTXVideoPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                cache_dir=VOLUME_CACHE,
                variant="fp16" # Use fp16 variant if available for speed/quality balance
            ).to("cuda")

            # 2. Image-to-Video Pipeline (shares components to save RAM)
            pipe_i2v = LTXImageToVideoPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                cache_dir=VOLUME_CACHE,
                variant="fp16",
                transformer=pipe_t2v.transformer,
                vae=pipe_t2v.vae,
                scheduler=pipe_t2v.scheduler,
                text_encoder=pipe_t2v.text_encoder,
                tokenizer=pipe_t2v.tokenizer
            ).to("cuda")
            
            print("--- LTX-VIDEO PIPELINES LOADED ---", flush=True)
        except Exception as e:
            print(f"FAILED to load model: {e}")
            raise e

def save_audiovisual_output(video_path, audio_tensor, sample_rate, output_path):
    """Merges video and audio into a final MP4."""
    try:
        # 1. Save Audio to temporary WAV
        temp_audio_path = output_path.replace(".mp4", "_audio.wav")
        
        # Normalize audio if needed
        audio_data = audio_tensor.cpu().numpy().squeeze()
        # Scale to 16-bit integer range if it's float [-1, 1]
        if audio_data.dtype == np.float32:
             audio_data = (audio_data * 32767).astype(np.int16)
             
        wav.write(temp_audio_path, sample_rate, audio_data)
        
        # 2. Merge using MoviePy
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(temp_audio_path)
        
        # Trim audio to match video length
        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.subclip(0, video_clip.duration)
            
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        
        # Cleanup
        video_clip.close()
        audio_clip.close()
        final_clip.close()
        os.remove(temp_audio_path)
        os.remove(video_path) # Remove silent video source
        
        return True
    except Exception as e:
        print(f"Audio Merge Failed: {e}")
        # If merge fails, return the silent video at least
        if os.path.exists(video_path):
            os.rename(video_path, output_path)
        return False

def handler(job):
    print(f"--- JOB START: {job['id']} ---", flush=True)
    if INIT_ERROR: return {"error": "Init failed", "traceback": INIT_ERROR}

    try:
        load_pipelines()
        
        job_input = job["input"]
        prompt = job_input.get("prompt", "A cinematic scene")
        negative_prompt = job_input.get("negative_prompt", "low quality, blurry, deformed, watermark")
        image_url = job_input.get("image_url", None)
        image_b64 = job_input.get("image_base64", None)
        
        # QUALITY SETTINGS
        # Increased steps for better quality (default was 40)
        num_inference_steps = int(job_input.get("num_inference_steps", 50)) 
        # Slightly higher guidance for sharper adherence
        guidance_scale = float(job_input.get("guidance_scale", 3.5))
        
        output_path = f"{OUTPUT_DIR}/{job['id']}.mp4"
        temp_video_path = f"{OUTPUT_DIR}/{job['id']}_silent.mp4"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Prepare Input Image
        input_image = None
        if image_url:
            input_image = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")
        elif image_b64:
            input_image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")

        # GENERATION
        with torch.no_grad():
            if input_image:
                print("Running Image-to-Video...", flush=True)
                # Resize image to be divisible by 32
                w, h = input_image.size
                w = w - (w % 32)
                h = h - (h % 32)
                input_image = input_image.resize((w, h))
                
                output = pipe_i2v(
                    image=input_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    output_type="pt" # Return tensors to handle audio manually if needed
                )
            else:
                print("Running Text-to-Video...", flush=True)
                # Standard resolution for LTX (768x512 is most stable)
                output = pipe_t2v(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=512, 
                    width=768,
                    num_frames=161, # ~6 seconds
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )

        # SAVE VIDEO
        # Access the video frames (it returns a VideoPipelineOutput)
        video_frames = output.frames[0]
        export_to_video(video_frames, temp_video_path, fps=24)
        
        # HANDLE AUDIO
        # LTX-Video generates audio if the prompt implies it, usually returned in .audios
        # Note: Current public LTX-Video pipeline in diffusers MIGHT NOT expose audio output directly 
        # depending on the version installed. 
        # If 'audios' attribute exists, we process it.
        
        has_audio = False
        if hasattr(output, "audios") and output.audios is not None:
             print("Audio generated! Merging...", flush=True)
             # Assumption: output.audios is a list of tensors/arrays
             audio_tensor = output.audios[0]
             # Default LTX sample rate is usually 16kHz or 24kHz
             sample_rate = getattr(output, "audio_sampling_rate", 16000)
             
             if save_audiovisual_output(temp_video_path, audio_tensor, sample_rate, output_path):
                 has_audio = True
        
        if not has_audio:
            print("No audio generated or audio merge skipped.", flush=True)
            if os.path.exists(temp_video_path):
                os.rename(temp_video_path, output_path)

        # Return Base64
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
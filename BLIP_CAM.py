import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
import torch
import logging
import time
from PIL import Image, ImageFont, ImageDraw
import sys
from threading import Thread, Lock
from queue import Queue
from gtts import gTTS
from playsound import playsound
import numpy as np
import os
import uuid

LANGUAGE_MODELS = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "te": "Helsinki-NLP/opus-mt-en-te",
    "ta": "Helsinki-NLP/opus-mt-en-ta",
    "kn": "Helsinki-NLP/opus-mt-en-kn",
    "en": None
}

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

class CaptionGenerator:
    def __init__(self, processor, model, device, target_lang="en"):
        self.processor = processor
        self.model = model
        self.device = device
        self.target_lang = target_lang
        self.current_caption = f"Initializing... ({device.upper()})"
        self.caption_queue = Queue(maxsize=1)
        self.lock = Lock()
        self.running = True
        self._init_translation_model()
        self.thread = Thread(target=self._caption_worker)
        self.thread.daemon = True
        self.thread.start()

    def _init_translation_model(self):
        if self.target_lang != "en":
            model_name = LANGUAGE_MODELS.get(self.target_lang)
            if model_name:
                self.translator_tokenizer = MarianTokenizer.from_pretrained(model_name)
                self.translator_model = MarianMTModel.from_pretrained(model_name)
                self.translator_model.to(self.device)
            else:
                self.translator_tokenizer = None
                self.translator_model = None
        else:
            self.translator_tokenizer = None
            self.translator_model = None

    def _caption_worker(self):
        while self.running:
            try:
                if not self.caption_queue.empty():
                    frame = self.caption_queue.get()
                    caption = self._generate_caption(frame)

                    if self.target_lang != "en":
                        caption = self._translate_caption(caption)

                    self._speak_caption(caption)

                    with self.lock:
                        self.current_caption = caption
            except Exception as e:
                logging.error(f"Caption worker error: {str(e)}")
            time.sleep(0.1)

    def _generate_caption(self, image):
        try:
            image_resized = cv2.resize(image, (640, 480))
            rgb_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=30,
                    num_beams=5,
                    num_return_sequences=1
                )
            caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            return caption
        except Exception as e:
            logging.error(f"Caption generation error: {str(e)}")
            return "Caption generation failed."

    def _translate_caption(self, caption):
        try:
            inputs = self.translator_tokenizer([caption], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            translated = self.translator_model.generate(**inputs, max_length=40)
            translated_text = self.translator_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            return translated_text.strip()
        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
            return caption

    def _speak_caption(self, caption):
        try:
            tts = gTTS(text=caption, lang=self.target_lang)
            filename = f"temp_{uuid.uuid4().hex}.mp3"
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            logging.error(f"TTS error: {str(e)}")

    def update_frame(self, frame):
        if self.caption_queue.empty():
            try:
                self.caption_queue.put_nowait(frame.copy())
            except:
                pass

    def get_caption(self):
        with self.lock:
            return self.current_caption

    def stop(self):
        self.running = False
        self.thread.join()

def get_gpu_usage():
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        memory_used_percent = (memory_allocated / memory_total) * 100
        return f"GPU Usage: {memory_used_percent:.2f}% | {memory_allocated:.2f} MB / {memory_total:.2f} MB"
    else:
        return "GPU not available"

def load_models():
    try:
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            torch.cuda.set_per_process_memory_fraction(0.9)
            blip_model = blip_model.to('cuda')
        return blip_processor, blip_model, device
    except Exception as e:
        logging.error(f"Model loading error: {str(e)}")
        return None, None, None

def live_stream_with_caption(processor, model, device, lang="en"):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Webcam not accessible.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    logger.info(f"Started webcam stream with BLIP captioning using {device.upper()}")
    caption_generator = CaptionGenerator(processor, model, device, target_lang=lang)

    prev_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read webcam frame.")
                break

            caption_generator.update_frame(frame)
            caption = caption_generator.get_caption()

            gpu_info = get_gpu_usage()
            fps = 1 / (time.time() - prev_time)
            prev_time = time.time()

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)

            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()

            y_offset = 40
            for line in [caption[i:i + 50] for i in range(0, len(caption), 50)]:
                draw.text((20, y_offset), line, font=font, fill=(0, 255, 0))
                y_offset += 30

            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            cv2.putText(frame, gpu_info, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            y_offset += 25
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            cv2.imshow("BLIP Live Captioning", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        caption_generator.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Loading BLIP model...")
    blip_processor, blip_model, device = load_models()
    if None in (blip_processor, blip_model):
        logger.error("Could not load BLIP model.")
        sys.exit(1)

    logger.info("Starting stream with caption translation and TTS...")

    # 🔽 Change to: "en", "hi", "te", "ta", "kn"
    selected_language = "hi"

    live_stream_with_caption(blip_processor, blip_model, device, lang=selected_language)

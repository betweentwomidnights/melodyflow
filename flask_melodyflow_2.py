from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import torch
import torchaudio
import time
import base64
import io
from audiocraft.models import MelodyFlow
import gc
import threading
from variations import VARIATIONS
import logging
import os
from contextlib import contextmanager
import json
import redis

import time
from audiocraft.models.loaders import load_compression_model, load_dit_model_melodyflow

def timed_get_pretrained(name: str = 'facebook/melodyflow-t24-30secs', device=None):
    """Instrumented version to see where time is spent."""
    if device is None:
        if torch.cuda.device_count():
            device = 'cuda'
        else:
            device = 'cpu'
    
    t0 = time.time()
    logger.info("Loading compression model...")
    compression_model = load_compression_model(name, device=device)
    t1 = time.time()
    logger.info(f"  → Compression model loaded in {t1-t0:.2f}s")
    
    logger.info("Removing weight norm...")
    def _remove_weight_norm(module):
        for m in module.modules():
            if hasattr(m, "conv") and hasattr(m.conv, "conv"):
                try:
                    torch.nn.utils.parametrize.remove_parametrizations(m.conv.conv, "weight")
                except: pass
            if hasattr(m, "convtr") and hasattr(m.convtr, "convtr"):
                try:
                    torch.nn.utils.parametrize.remove_parametrizations(m.convtr.convtr, "weight")
                except: pass
    
    _remove_weight_norm(compression_model)
    compression_model.to(device)
    t2 = time.time()
    logger.info(f"  → Weight norm + device transfer in {t2-t1:.2f}s")
    
    logger.info("Loading DiT model...")
    lm = load_dit_model_melodyflow(name, device=device)
    t3 = time.time()
    logger.info(f"  → DiT model loaded in {t3-t2:.2f}s")
    
    logger.info(f"TOTAL model load time: {t3-t0:.2f}s")
    
    return MelodyFlow(
        name=name,
        compression_model=compression_model,
        lm=lm,
        # target_device=device,        # <-- add this
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('melodyflow')

app = Flask(__name__)
app.start_time = time.time()
CORS(app)

# Redis setup - shared with g4lwebsockets
redis_client = redis.StrictRedis(host='redis', port=6379, db=0)

# Use gevent for WebSocket support
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='gevent',
    message_queue='redis://redis:6379',  # Shared Redis with g4lwebsockets
    logger=True,
    engineio_logger=False,
)

# Global variables
model = None
model_last_used = None
model_cleanup_timer = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_TIMEOUT = 1  # 10 minutes to keep model warm


class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""

    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


@contextmanager
def resource_cleanup():
    """Context manager for GPU cleanup - simplified version."""

    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def schedule_model_cleanup():
    """Schedule model cleanup after timeout."""

    global model_cleanup_timer
    if model_cleanup_timer:
        model_cleanup_timer.cancel()

    def cleanup():
        global model, model_last_used
        if model and model_last_used and time.time() - model_last_used > MODEL_TIMEOUT:
            logger.info("Cleaning up MelodyFlow model after timeout...")
            del model
            model = None
            with resource_cleanup():
                pass

    model_cleanup_timer = threading.Timer(MODEL_TIMEOUT, cleanup)
    model_cleanup_timer.start()


def load_audio_from_file(file_path: str, target_sr: int = 32000) -> torch.Tensor:
    """Load and preprocess audio from file path."""

    try:
        if not os.path.exists(file_path):
            raise AudioProcessingError(f"Audio file not found: {file_path}")

        waveform, sr = torchaudio.load(file_path)

        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)

        # Ensure stereo
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        return waveform.unsqueeze(0).to(DEVICE)

    except Exception as e:
        raise AudioProcessingError(f"Failed to load audio from file: {str(e)}")


def load_audio_from_base64(audio_base64: str, target_sr: int = 32000) -> torch.Tensor:
    """Load and preprocess audio from base64 string."""

    try:
        audio_data = base64.b64decode(audio_base64)
        audio_file = io.BytesIO(audio_data)
        waveform, sr = torchaudio.load(audio_file)

        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)

        # Ensure stereo
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        return waveform.unsqueeze(0).to(DEVICE)

    except Exception as e:
        raise AudioProcessingError(f"Failed to load audio: {str(e)}")


def save_audio_to_base64(waveform: torch.Tensor, sample_rate: int = 32000) -> str:
    """Convert audio tensor to base64 string."""

    try:
        buffer = io.BytesIO()
        torchaudio.save(buffer, waveform.cpu(), sample_rate, format="wav")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        raise AudioProcessingError(f"Failed to save audio: {str(e)}")


def find_max_duration(
    model: MelodyFlow,
    waveform: torch.Tensor,
    sr: int = 32000,
    max_token_length: int = 750,
) -> tuple:
    """Binary search to find maximum duration that produces tokens under the limit."""

    min_seconds = 1
    max_seconds = waveform.shape[-1] / sr
    best_duration = min_seconds
    best_tokens = None

    while max_seconds - min_seconds > 0.1:
        mid_seconds = (min_seconds + max_seconds) / 2
        samples = int(mid_seconds * sr)
        test_waveform = waveform[..., :samples]

        try:
            tokens = model.encode_audio(test_waveform.to(model.device))
            token_length = tokens.shape[-1]

            if token_length <= max_token_length:
                best_duration = mid_seconds
                best_tokens = tokens
                min_seconds = mid_seconds
            else:
                max_seconds = mid_seconds

        except Exception:
            max_seconds = mid_seconds

    return best_duration, best_tokens


def process_audio(
    waveform: torch.Tensor,
    variation_name: str,
    session_id: str,
    custom_flowstep: float = None,
    solver: str = "euler",
    custom_prompt: str = None,
    progress_callback=None,
) -> torch.Tensor:
    """Process audio with selected variation."""

    global model, model_last_used

    try:
        if variation_name not in VARIATIONS:
            raise AudioProcessingError(f"Unknown variation: {variation_name}")

        config = VARIATIONS[variation_name].copy()
        flowstep = custom_flowstep if custom_flowstep is not None else config['default_flowstep']

        if custom_prompt is not None:
            config['prompt'] = custom_prompt

        with resource_cleanup():
            # Store warming status for HTTP clients BEFORE loading model
            try:
                redis_client.setex(
                    f"status:{session_id}", 
                    3600, 
                    json.dumps({'status': 'warming', 'timestamp': time.time()})
                )
            except Exception as e:
                logger.warning(f"Failed to store warming status in Redis: {e}")

            # Load or use cached model
            if model is None:
                logger.info("Loading MelodyFlow model...")
                
                # Emit warming status once for WebSocket clients
                try:
                    socketio.emit(
                        'queue_status',
                        {
                            'status': 'warming',
                            'message': 'Loading MelodyFlow model...',
                            'session_id': session_id,
                        },
                        room=session_id,
                        namespace='/',
                    )
                except Exception as e:
                    logger.warning(f"Failed to emit warming status: {e}")
                
                # Load the model (this takes time)
                model = timed_get_pretrained('facebook/melodyflow-t24-30secs', device=DEVICE)
                
                logger.info("MelodyFlow model loaded")
            else:
                logger.info("Using cached MelodyFlow model")

            # Update last used time and schedule cleanup
          #  model_last_used = time.time()
          #  schedule_model_cleanup()

            # Find valid duration and get tokens
            max_valid_duration, tokens = find_max_duration(model, waveform)
            config['duration'] = max_valid_duration

            # Override steps and regularization based on solver
            if solver.lower() == "midpoint":
                steps = 64
                use_regularize = False
            else:
                steps = config['steps']
                use_regularize = True

            # Set model parameters
            model.set_generation_params(
                solver=solver.lower(),
                steps=steps,
                duration=config['duration'],
            )

            model.set_editing_params(
                solver=solver.lower(),
                steps=steps,
                target_flowstep=flowstep,
                regularize=use_regularize,
                regularize_iters=2 if use_regularize else 0,
                keep_last_k_iters=1 if use_regularize else 0,
                lambda_kl=0.2 if use_regularize else 0.0,
            )

            # Only *after* all of that succeeds:
            model_last_used = time.time()
            schedule_model_cleanup()

            # Track if we've transitioned to processing status
            processing_status_set = False

            if progress_callback:
                def model_progress_callback(elapsed_steps: int, total_steps: int):
                    nonlocal processing_status_set
                    
                    progress_percent = min((elapsed_steps / total_steps) * 100, 99.9)
                    
                    # Store progress for HTTP polling clients
                    try:
                        redis_client.setex(f"progress:{session_id}", 3600, str(int(progress_percent)))
                    except Exception as e:
                        logger.warning(f"Failed to store progress in Redis: {e}")
                    
                    # Emit progress for WebSocket clients
                    try:
                        socketio.emit(
                            'progress_update',
                            {
                                'progress': round(progress_percent, 2),
                                'session_id': session_id,
                            },
                            room=session_id,
                            namespace='/',
                        )
                    except Exception as e:
                        logger.warning(f"Failed to emit progress: {e}")
                    
                    # On first progress, transition from warming to processing
                    if not processing_status_set and progress_percent > 0:
                        try:
                            redis_client.setex(
                                f"status:{session_id}",
                                3600,
                                json.dumps({'status': 'processing', 'timestamp': time.time()})
                            )
                            processing_status_set = True
                        except Exception as e:
                            logger.warning(f"Failed to update processing status in Redis: {e}")
                    
                    # Also call the original callback if provided
                    progress_callback(elapsed_steps, total_steps)

                model._progress_callback = model_progress_callback
            else:
                # Even without external callback, we need to emit progress
                def model_progress_callback(elapsed_steps: int, total_steps: int):
                    nonlocal processing_status_set
                    
                    progress_percent = min((elapsed_steps / total_steps) * 100, 99.9)
                    
                    # Store progress for HTTP polling clients
                    try:
                        redis_client.setex(f"progress:{session_id}", 3600, str(int(progress_percent)))
                    except Exception as e:
                        logger.warning(f"Failed to store progress in Redis: {e}")
                    
                    # Emit progress for WebSocket clients
                    try:
                        socketio.emit(
                            'progress_update',
                            {
                                'progress': round(progress_percent, 2),
                                'session_id': session_id,
                            },
                            room=session_id,
                            namespace='/',
                        )
                    except Exception as e:
                        logger.warning(f"Failed to emit progress: {e}")
                    
                    # On first progress, transition from warming to processing
                    if not processing_status_set and progress_percent > 0:
                        try:
                            redis_client.setex(
                                f"status:{session_id}",
                                3600,
                                json.dumps({'status': 'processing', 'timestamp': time.time()})
                            )
                            processing_status_set = True
                        except Exception as e:
                            logger.warning(f"Failed to update processing status in Redis: {e}")
                
                model._progress_callback = model_progress_callback

            edited_audio = model.edit(
                prompt_tokens=tokens,
                descriptions=[config['prompt']],
                src_descriptions=[""],
                progress=True,
                return_tokens=True,
            )

            return edited_audio[0][0]

            

    except Exception as e:
        raise AudioProcessingError(f"Failed to process audio: {str(e)}")


@app.route('/transform', methods=['POST'])
def transform_audio():
    """Handle audio transformation requests - simplified, no queueing."""

    try:
        data = request.get_json()
        if not data or 'variation' not in data or 'session_id' not in data:
            return jsonify({'error': 'Missing required data'}), 400

        has_audio_data = 'audio' in data and data['audio']
        has_file_path = 'audio_file_path' in data and data['audio_file_path']

        if not has_audio_data and not has_file_path:
            return jsonify({'error': 'No audio data or file path provided'}), 400

        custom_flowstep = data.get('flowstep')
        if custom_flowstep is not None:
            try:
                custom_flowstep = float(custom_flowstep)
                if custom_flowstep <= 0:
                    return jsonify({'error': 'Flowstep must be positive'}), 400
            except ValueError:
                return jsonify({'error': 'Invalid flowstep value'}), 400

        solver = data.get('solver', 'euler')
        if solver not in ['euler', 'midpoint']:
            return jsonify({'error': 'Invalid solver'}), 400

        custom_prompt = data.get('custom_prompt')
        session_id = data['session_id']

        if has_file_path:
            input_waveform = load_audio_from_file(data['audio_file_path'])
        else:
            input_waveform = load_audio_from_base64(data['audio'])

        def progress_callback(current, total):
            # This is called by process_audio's internal callback
            # which already handles Redis storage and SocketIO emission
            socketio.sleep(0)

        processed_waveform = process_audio(
            input_waveform,
            data['variation'],
            session_id,
            custom_flowstep,
            solver,
            custom_prompt=custom_prompt,
            progress_callback=progress_callback,
        )

        output_base64 = save_audio_to_base64(processed_waveform)

        del input_waveform
        del processed_waveform

        # Emit 100% progress
        try:
            redis_client.setex(f"progress:{session_id}", 3600, "100")
        except:
            pass
        
        try:
            socketio.emit(
                'progress_update',
                {
                    'progress': 100.0,
                    'session_id': session_id,
                },
                room=session_id,
                namespace='/',
            )
        except:
            pass
        
        socketio.sleep(0)

        return jsonify(
            {
                'audio': output_base64,
                'message': 'Audio processed successfully',
            }
        )

    except AudioProcessingError as e:
        logger.error(f"Audio processing error: {str(e)}")
        return jsonify({'error': str(e)}), e.status_code
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


@app.route('/variations', methods=['GET'])
def get_variations():
    """Return list of available variations."""

    try:
        variations_with_details = {
            name: {
                'prompt': VARIATIONS[name]['prompt'],
                'flowstep': VARIATIONS[name]['flowstep'],
            }
            for name in VARIATIONS.keys()
        }
        return jsonify({'variations': variations_with_details})
    except Exception as e:
        return jsonify({'error': f'Failed to fetch variations: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""

    try:
        gpu_available = torch.cuda.is_available()
        model_loaded = model is not None

        status = {
            'status': 'healthy' if gpu_available else 'degraded',
            'gpu_available': gpu_available,
            'model_loaded': model_loaded,
            'uptime': time.time() - app.start_time,
        }

        return jsonify(status), 200 if status['status'] == 'healthy' else 503

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503


@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')


if __name__ == '__main__':
    logger.info("Starting MelodyFlow service with gevent...")
    socketio.run(app, host='0.0.0.0', port=8002)
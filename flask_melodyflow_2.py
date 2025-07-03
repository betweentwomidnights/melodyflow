from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import torch
import torchaudio
import time
import base64
import io
from audiocraft.models import MelodyFlow
import gc
import threading
from variations import VARIATIONS
import psutil
import logging
from logging.handlers import RotatingFileHandler
import traceback
import os
import sys

# Add these imports at the top
import weakref
# Add these imports and keep existing ones
from contextlib import contextmanager

from redis import Redis
from rq import Queue

from typing import Callable, Optional

# Connect to Redis db=1 for job queue
job_redis = Redis(host='redis', port=6379, db=1)
task_queue = Queue('melodyflow', connection=job_redis)

# Add these global variables
model_ref = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model_last_used = None
model_cleanup_timer = None
MODEL_TIMEOUT = 300  # 5 minutes - adjust as needed

def schedule_model_cleanup():
    """Schedule model cleanup after timeout."""
    global model_cleanup_timer
    if model_cleanup_timer:
        model_cleanup_timer.cancel()
    
    def cleanup():
        global model, model_last_used
        if model and model_last_used and time.time() - model_last_used > MODEL_TIMEOUT:
            print("🧹 Cleaning up MelodyFlow model after timeout...")
            del model
            model = None
            with resource_cleanup():
                pass
    
    model_cleanup_timer = threading.Timer(MODEL_TIMEOUT, cleanup)
    model_cleanup_timer.start()

@contextmanager
def resource_cleanup():
    """Context manager to ensure proper cleanup of GPU resources."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

# Configure logging to filter out the specific error
class WebSocketErrorFilter(logging.Filter):
    def filter(self, record):
        return 'Cannot obtain socket from WSGI environment' not in str(record.msg)

# Add the filter to the root logger
logging.getLogger().addFilter(WebSocketErrorFilter())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('melodyflow')
handler = RotatingFileHandler('melodyflow.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
logger.addHandler(handler)

app = Flask(__name__)
app.start_time = time.time()  # Add this line right after app initialization
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_lock = threading.Lock()  # Lock for model access

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code

# Add this new function to load audio from file path
def load_audio_from_file(file_path: str, target_sr: int = 32000) -> torch.Tensor:
    """Load and preprocess audio from file path."""
    try:
        if not os.path.exists(file_path):
            raise AudioProcessingError(f"Audio file not found: {file_path}")
        
        # Load audio directly from file
        waveform, sr = torchaudio.load(file_path)

        # Resample if needed
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)

        # Ensure stereo
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        return waveform.unsqueeze(0).to(device)

    except Exception as e:
        raise AudioProcessingError(f"Failed to load audio from file: {str(e)}")

def queued_transform(audio_data_or_path, variation, session_id, custom_flowstep=None, solver="euler", custom_prompt=None, is_file_path=False):
    """Wrapper function for process_audio to be used by RQ."""
    try:
        with resource_cleanup():
            # Load audio based on whether we have file path or base64
            if is_file_path:
                input_waveform = load_audio_from_file(audio_data_or_path)
            else:
                input_waveform = load_audio_from_base64(audio_data_or_path)

            # Create progress callback that emits to the correct session
            def progress_callback(current, total):
                progress = min((current / total) * 100, 99.9)
                socketio.emit('progress', {
                    'progress': round(progress, 2),
                    'session_id': session_id
                })

            # Process with progress updates
            processed_waveform = process_audio(
                input_waveform, 
                variation, 
                session_id,
                custom_flowstep,
                solver,
                custom_prompt=custom_prompt,
                progress_callback=progress_callback
            )
            
            # Send 100% progress after processing
            socketio.emit('progress', {
                'progress': 100.0,
                'session_id': session_id
            })

            output_base64 = save_audio_to_base64(processed_waveform)
            
            # Explicitly delete intermediate tensors
            del input_waveform
            del processed_waveform

            # Emit completion event
            socketio.emit('transform_complete', {
                'session_id': session_id,
                'status': 'success'
            })

            return {
                'audio': output_base64,
                'message': 'Audio processed successfully'
            }
    except Exception as e:
        # Emit error event
        socketio.emit('transform_error', {
            'session_id': session_id,
            'error': str(e)
        })
        raise AudioProcessingError(str(e))
    
def process_immediate(data, custom_flowstep, solver="euler", custom_prompt=None):
    """Process transformation immediately with progress updates."""
    with resource_cleanup():
        # Check if we have file path or base64 data
        if 'audio_file_path' in data and data['audio_file_path']:
            print(f"Loading audio from file: {data['audio_file_path']}")
            input_waveform = load_audio_from_file(data['audio_file_path'])
        elif 'audio' in data:
            print("Loading audio from base64 data")
            input_waveform = load_audio_from_base64(data['audio'])
        else:
            raise AudioProcessingError("No audio data or file path provided")
        
        def progress_callback(current, total):
            progress = min((current / total) * 100, 99.9)
            socketio.emit('progress', {
                'progress': round(progress, 2),
                'session_id': data['session_id']
            })

        processed_waveform = process_audio(
            input_waveform, 
            data['variation'], 
            data['session_id'],
            custom_flowstep,
            solver,
            custom_prompt=custom_prompt,
            progress_callback=progress_callback
        )
        
        output_base64 = save_audio_to_base64(processed_waveform)
        
        # Cleanup
        del input_waveform
        del processed_waveform
        
        # Send 100% progress
        socketio.emit('progress', {
            'progress': 100.0,
            'session_id': data['session_id']
        })

        return jsonify({
            'audio': output_base64,
            'message': 'Audio processed successfully'
        })

def get_system_resources():
    """Monitor system resources."""
    try:
        # CPU and RAM
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        # GPU memory using torch
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**2  # Convert to MB

        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'gpu_memory_used_mb': gpu_memory_used
        }
    except Exception as e:
        logger.error(f"Error getting system resources: {e}")
        return None

def check_resource_availability():
    """Check if enough resources are available for processing."""
    resources = get_system_resources()
    if resources:
        # Define thresholds
        if (resources['cpu_percent'] > 90 or
            resources['memory_percent'] > 90 or
            resources['gpu_memory_used_mb'] > 10000):  # 10GB threshold
            raise AudioProcessingError(
                "Server is currently at capacity. Please try again later.",
                status_code=503
            )

@app.before_request
def before_request():
    """Check resources before processing requests."""
    if request.endpoint != 'health_check':  # Skip for health checks
        check_resource_availability()

@app.errorhandler(AudioProcessingError)
def handle_audio_processing_error(error):
    logger.error(f"Audio processing error: {str(error)}\n{traceback.format_exc()}")
    return jsonify({'error': str(error)}), error.status_code

@app.errorhandler(Exception)
def handle_generic_error(error):
    logger.error(f"Unexpected error: {str(error)}\n{traceback.format_exc()}")
    return jsonify({'error': 'An unexpected error occurred'}), 500

# Audio Processing

def load_model():
    """Initialize the MelodyFlow model with improved memory management."""
    global model, model_ref
    with model_lock:
        if model is None:
            print("Loading MelodyFlow model...")
            model = MelodyFlow.get_pretrained('facebook/melodyflow-t24-30secs', device=DEVICE)
            # Create weak reference to track model
            model_ref = weakref.ref(model)
    return model

def load_audio_from_base64(audio_base64: str, target_sr: int = 32000) -> torch.Tensor:
    """Load and preprocess audio from base64 string."""
    try:
        # Decode base64 to binary
        audio_data = base64.b64decode(audio_base64)
        audio_file = io.BytesIO(audio_data)

        # Load audio
        waveform, sr = torchaudio.load(audio_file)

        # Resample if needed
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)

        # Ensure stereo
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        return waveform.unsqueeze(0).to(device)

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

def process_audio(waveform: torch.Tensor, variation_name: str, session_id: str, 
               custom_flowstep: float = None, solver: str = "euler", 
               custom_prompt: str = None, progress_callback: Callable = None) -> torch.Tensor:
    """Process audio with selected variation using scoped model instantiation.
    
    Args:
        waveform: Input audio tensor
        variation_name: Name of the variation from VARIATIONS
        session_id: Unique session identifier
        custom_flowstep: Optional custom flowstep parameter
        solver: Solver type ("euler" or "midpoint")
        custom_prompt: Optional custom prompt to override variation prompt
        progress_callback: Optional callback for progress updates
    """
    global model, model_last_used  # Add this line

    # DEBUG: Check cache locations
    print(f"🔍 HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
    print(f"🔍 TORCH_HOME: {os.environ.get('TORCH_HOME', 'not set')}")
    print(f"🔍 Current working directory: {os.getcwd()}")
    print(f"🔍 Home directory: {os.path.expanduser('~')}")

    cache_locations = [
        "/app/.cache",
        "/root/.cache", 
        "/app/.cache/huggingface",
        "/root/.cache/huggingface",
        "/root/.cache/torch",
        "/app/.cache/torch"
    ]

    for loc in cache_locations:
        exists = os.path.exists(loc)
        print(f"🔍 {loc}: {'EXISTS' if exists else 'NOT FOUND'}")
        if exists:
            try:
                files = os.listdir(loc)
                print(f"    Contents: {files}")
            except:
                print("    (permission denied)")
    
    try:
        if variation_name not in VARIATIONS:
            raise AudioProcessingError(f"Unknown variation: {variation_name}")
        
        config = VARIATIONS[variation_name].copy()
        flowstep = custom_flowstep if custom_flowstep is not None else config['default_flowstep']
        
        if custom_prompt is not None:
            config['prompt'] = custom_prompt

        with resource_cleanup():
            # Use global model with timeout-based cleanup
            if model is None:
                print("🔄 Loading MelodyFlow model...")
                model = MelodyFlow.get_pretrained('facebook/melodyflow-t24-30secs', device=device)
            else:
                print("✅ Using cached MelodyFlow model")
            
            # Update last used time and schedule cleanup
            model_last_used = time.time()
            schedule_model_cleanup()

            # Find valid duration and get tokens
            max_valid_duration, tokens = find_max_duration(model, waveform)
            config['duration'] = max_valid_duration

            # Override steps and regularization based on solver
            if solver.lower() == "midpoint":
                steps = 64  # Fixed steps for midpoint
                use_regularize = False  # Disable regularization for midpoint
            else:  # Default to euler
                steps = config['steps']  # Use steps from variation config
                use_regularize = True    # Enable regularization for euler

            # Set model parameters
            model.set_generation_params(
                solver=solver.lower(),  # Use the specified solver
                steps=steps,            # Use the determined steps value
                duration=config['duration'],
            )

            model.set_editing_params(
                solver=solver.lower(),  # Use the specified solver
                steps=steps,            # Use the determined steps value
                target_flowstep=flowstep,
                regularize=use_regularize,  # Conditionally enable/disable regularization
                regularize_iters=2 if use_regularize else 0,
                keep_last_k_iters=1 if use_regularize else 0,
                lambda_kl=0.2 if use_regularize else 0.0,
            )

            # Only set progress callback if provided
            if progress_callback:
                def model_progress_callback(elapsed_steps: int, total_steps: int):
                    progress_callback(elapsed_steps, total_steps)
                model._progress_callback = model_progress_callback

            edited_audio = model.edit(
                prompt_tokens=tokens,
                descriptions=[config['prompt']],
                src_descriptions=[""],
                progress=True,
                return_tokens=True
            )

            return edited_audio[0][0]

    except Exception as e:
        raise AudioProcessingError(f"Failed to process audio: {str(e)}")

def find_max_duration(model: MelodyFlow, waveform: torch.Tensor, sr: int = 32000, max_token_length: int = 750) -> tuple:
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
            tokens = model.encode_audio(test_waveform)
            token_length = tokens.shape[-1]

            if token_length <= max_token_length:
                best_duration = mid_seconds
                best_tokens = tokens
                min_seconds = mid_seconds
            else:
                max_seconds = mid_seconds

        except Exception as e:
            max_seconds = mid_seconds

    return best_duration, best_tokens

@app.route('/transform', methods=['POST'])
def transform_audio():
    """Handle audio transformation requests with job queue and progress updates."""
    try:
        data = request.get_json()
        if not data or 'variation' not in data or 'session_id' not in data:
            return jsonify({'error': 'Missing required data'}), 400
        
        # Check if we have audio data or file path
        has_audio_data = 'audio' in data and data['audio']
        has_file_path = 'audio_file_path' in data and data['audio_file_path']
        
        if not has_audio_data and not has_file_path:
            return jsonify({'error': 'No audio data or file path provided'}), 400

        # Validate flowstep
        custom_flowstep = data.get('flowstep')
        if custom_flowstep is not None:
            try:
                custom_flowstep = float(custom_flowstep)
                if custom_flowstep <= 0:
                    return jsonify({'error': 'Flowstep must be positive'}), 400
            except ValueError:
                return jsonify({'error': 'Invalid flowstep value'}), 400
        
        # Get solver parameter (default to "euler" if not provided)
        solver = data.get('solver', 'euler')
        if solver not in ['euler', 'midpoint']:
            return jsonify({'error': 'Invalid solver. Must be "euler" or "midpoint"'}), 400

        # Get custom_prompt if provided
        custom_prompt = data.get('custom_prompt')
            
        session_id = data['session_id']

        # Check if there are any jobs in the queue
        queue_length = len(task_queue)
        if queue_length > 0:
            # Emit queue position update
            socketio.emit('queue_position', {
                'position': queue_length + 1,
                'session_id': session_id
            })

            # Determine what to pass to the queue
            if has_file_path:
                audio_data_or_path = data['audio_file_path']
                is_file_path = True
            else:
                audio_data_or_path = data['audio']
                is_file_path = False

            # Enqueue the job
            job = task_queue.enqueue(
                queued_transform,
                args=(audio_data_or_path, data['variation'], session_id),
                kwargs={
                    'custom_flowstep': custom_flowstep,
                    'solver': solver,
                    'custom_prompt': custom_prompt,
                    'is_file_path': is_file_path  # NEW: Tell the worker how to handle the data
                },
                job_timeout='10m',
                result_ttl=600,
                on_success=lambda job, connection, result: 
                    socketio.emit('job_complete', {
                        'session_id': session_id,
                        'job_id': job.id
                    }),
                on_failure=lambda job, connection, type, value, traceback: 
                    socketio.emit('job_failed', {
                        'session_id': session_id,
                        'job_id': job.id,
                        'error': str(value)
                    })
            )

            return jsonify({
                'status': 'queued',
                'job_id': job.id,
                'position': queue_length + 1,
                'session_id': session_id
            }), 202
        else:
            # If queue is empty, process immediately
            return process_immediate(data, custom_flowstep, solver, custom_prompt)

    except AudioProcessingError as e:
        socketio.emit('transform_error', {
            'session_id': data.get('session_id'),
            'error': str(e)
        })
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        socketio.emit('transform_error', {
            'session_id': data.get('session_id'),
            'error': str(e)
        })
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/transform/status/<job_id>', methods=['GET'])
def check_job_status(job_id):
    """Check the status of a transformation job."""
    try:
        print(f"Received status check for job {job_id}")
        job = task_queue.fetch_job(job_id)
        
        if job is None:
            print(f"Job {job_id} not found")
            return jsonify({'error': 'Job not found'}), 404

        if job.is_finished:
            print(f"Job {job_id} is finished")
            # If job is complete, return the audio data
            result = job.result
            print(f"Returning result for job {job_id}")
            return jsonify(result)
        elif job.is_failed:
            print(f"Job {job_id} failed with error: {job.exc_info}")
            return jsonify({
                'error': str(job.exc_info),
                'status': 'failed'
            }), 400
        else:
            status = job.get_status()
            position = len(task_queue)
            enqueued_at = job.enqueued_at.isoformat() if job.enqueued_at else None
            started_at = job.started_at.isoformat() if job.started_at else None
            
            response_data = {
                'status': status,
                'position': position,
                'enqueued_at': enqueued_at,
                'started_at': started_at,
            }
            
            print(f"Job {job_id} status: {response_data}")
            return jsonify(response_data), 202

    except Exception as e:
        print(f"Error checking job status for {job_id}: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@app.route('/variations', methods=['GET'])
def get_variations():
    """Return list of available variations with CORS support."""
    try:
        variations_list = list(VARIATIONS.keys())
        variations_with_details = {
            name: {
                'prompt': VARIATIONS[name]['prompt'],
                'flowstep': VARIATIONS[name]['flowstep']
            } for name in variations_list
        }
        return jsonify({
            'variations': variations_with_details
        })
    except Exception as e:
        return jsonify({'error': f'Failed to fetch variations: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint with detailed system status."""
    try:
        resources = get_system_resources()
        model_loaded = model is not None
        gpu_available = torch.cuda.is_available()

        status = {
            'status': 'healthy' if model_loaded and gpu_available else 'degraded',
            'gpu_available': gpu_available,
            'model_loaded': model_loaded,
            'system_resources': resources,
            'uptime': time.time() - app.start_time,
            'version': os.getenv('MELODYFLOW_VERSION', 'dev'),
            'environment': os.getenv('FLASK_ENV', 'production')
        }

        return jsonify(status), 200 if status['status'] == 'healthy' else 503

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503
    
@app.route('/cleanup', methods=['POST'])
def manual_cleanup():
    """Manually cleanup model if needed."""
    global model
    if model:
        del model
        model = None
        with resource_cleanup():
            pass
        return jsonify({'message': 'Model cleaned up successfully'})
    return jsonify({'message': 'No model to cleanup'})

if __name__ == '__main__':
    # socketio.run(app, debug=False, host='0.0.0.0', port=8002)

    # Production mode with gevent-websocket (uncomment for production)
    # from gevent import pywsgi
    # from geventwebsocket.handler import WebSocketHandler
    # server = pywsgi.WSGIServer(('0.0.0.0', 8002), app, handler_class=WebSocketHandler)
    # server.serve_forever()

    # Production mode with waitress (currently shows websocket errors but everything works fine bro)
    
    from waitress import serve
    
    
    print("Starting MelodyFlow service in production mode...")
    print("Note: You may see a WebSocket environment message - this can be safely ignored as all functionality works correctly.")
    
    # Redirect stderr to filter out the specific error
    class StderrFilter:
        def write(self, text):
            if 'Cannot obtain socket from WSGI environment' not in text:
                sys.__stderr__.write(text)
        def flush(self):
            sys.__stderr__.flush()
    
    sys.stderr = StderrFilter()
    
    serve(app, host='0.0.0.0', port=8002, threads=4)

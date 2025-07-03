# melodyflow

flask service for audio transformation using meta's melodyflow model.

this service transforms input audio while retaining bpm and chord structure. great for turning guitar riffs into orchestral pieces or cleaning up noisy musicgen continuations.

## what it does

- **audio transformation**: takes input audio and transforms it based on text prompts or presets
- **style preservation**: maintains bpm and chord structure during transformation  
- **preset variations**: predefined transformations in `variations.py` (still figuring out what works best)
- **custom prompts**: supports custom text-based transformations
- **progress tracking**: real-time progress updates via websockets

## audiocraft integration

this container includes a modified audiocraft repository pulled from:
https://huggingface.co/spaces/facebook/melodyflow

this version seems to have breaking changes to musicgen functionality, which is why melodyflow requires its own container separate from the main gary backend. it might also explain why meta hasn't integrated melodyflow into the official audiocraft repository yet.

## building

```bash
docker build -t thecollabagepatch/melodyflow:latest -f Dockerfile.melodyflow .
```

this specific tag works with the docker-compose.yml in the main [gary-backend-combined](https://github.com/betweentwomidnights/gary-backend-combined) repo.

## example usage

the service accepts audio transformation requests:

```bash
# basic transformation with preset variation
curl -X POST http://localhost:8002/transform \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "base64_encoded_audio_data",
    "variation": "strings_quartet",
    "session_id": "test_session"
  }'

# custom prompt transformation
curl -X POST http://localhost:8002/transform \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "base64_encoded_audio_data", 
    "variation": "custom",
    "custom_prompt": "808 bass",
    "session_id": "test_session"
  }'
```

## variations

predefined transformation presets are configured in `variations.py`. each variation includes:
- text prompt for transformation style
- flowstep parameter for transformation strength
- steps for diffusion process

## queue management

this service includes internal RQ (redis queue) worker functionality for job management. in the main gary backend, queue management is handled by the separate `gpu-queue-service` now, but we've left the rq worker in there because it doesn't break anything, and leaves the possibility to run this service by itself at some point.

## integration with gary backend

in the gary ecosystem:
- transforms noisy musicgen continuations into cool shit
- enables style transfer between different musical genres
- maintains tempo alignment with ableton projects
- works alongside the main websockets backend via the queue service

## parameters

- **variation**: preset name from variations.py or "custom"
- **custom_prompt**: text description for custom transformations
- **flowstep**: transformation strength (0.0-1.0, higher = more transformation)
- **solver**: diffusion solver ("euler" or "midpoint")
- **session_id**: unique identifier for progress tracking

## health monitoring

check service status:
```bash
curl http://localhost:8002/health
```

returns system resources, model status, and gpu availability.

## why separate container

this service requires its own container because:
1. contains modified audiocraft with breaking changes to musicgen
2. different dependency requirements from main backend
3. can be scaled independently for heavy transformation workloads
4. allows for future standalone deployment scenarios

## development

the flask service runs on port 8002 and includes:
- websocket support for real-time progress updates
- comprehensive error handling and logging
- gpu memory management and cleanup
- resource monitoring and capacity checking

based on meta's melodyflow: https://huggingface.co/spaces/facebook/melodyflow
from typing import Dict, Any, Optional
import aiohttp
import secrets
import os
from dotenv import load_dotenv

load_dotenv()

class APIError(Exception):
    """Raised when the API returns an error response."""
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code

class ImageGen:
    def __init__(self, api_key: str, base_url: str = os.getenv("HEURIST_SEQUENCER_URL")):
        self.api_key = api_key
        self.base_url = base_url
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        await self._create_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._close_session()

    async def _create_session(self):
        """Create aiohttp session if it doesn't exist."""
        if self._session is None:
            self._session = aiohttp.ClientSession(headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })

    async def _close_session(self):
        """Close the aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _ensure_session(self):
        """Ensure session exists before making requests."""
        if self._session is None:
            await self._create_session()

    async def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            await self._ensure_session()
            
            # Generate a random job ID
            job_id = f"sdk-image-{secrets.token_hex(5)}"

            # Extract parameters
            prompt = params.get('prompt', '')
            neg_prompt = params.get('neg_prompt')
            num_iterations = params.get('num_iterations')
            guidance_scale = params.get('guidance_scale')
            width = params.get('width')
            height = params.get('height')
            seed = params.get('seed')
            model = params.get('model')
            job_id_prefix = params.get('job_id_prefix', 'sdk-image')

            # Handle special model cases
            if model == 'Zeek':
                prompt = prompt.replace('Zeek', 'z33k').replace('zeek', 'z33k')
            elif model == 'Philand':
                prompt = prompt.replace('Philand', 'ph1land').replace('philand', 'ph1land')

            # Prepare model input
            model_input = {
                "prompt": prompt
            }
            if neg_prompt:
                model_input["neg_prompt"] = neg_prompt
            if num_iterations:
                model_input["num_iterations"] = num_iterations
            if guidance_scale:
                model_input["guidance_scale"] = guidance_scale
            if width:
                model_input["width"] = width
            if height:
                model_input["height"] = height
            if seed:
                # Handle large seed values
                seed_int = int(seed)
                if seed_int > 9007199254740991:  # Number.MAX_SAFE_INTEGER
                    seed_int = seed_int % 9007199254740991
                model_input["seed"] = seed_int

            # Prepare the full request parameters
            request_params = {
                "job_id": f"{job_id_prefix}-{secrets.token_hex(5)}",
                "model_input": {
                    "SD": model_input
                },
                "model_type": "SD",
                "model_id": model,
                "deadline": 30,
                "priority": 1
            }

            async with self._session.post(
                f"{self.base_url}/submit_job",
                json=request_params
            ) as response:
                if not response.ok:
                    if str(response.status).startswith(('4', '5')):
                        raise APIError("Generate image error. Please try again later")
                    raise APIError(f"HTTP error! status: {response.status}")

                url = await response.text()
                url = url.strip('"')  # Remove quotes if present

                return {
                    "url": url,
                    "model": model,
                    **model_input
                }

        except Exception as e:
            if isinstance(e, APIError):
                raise e
            raise APIError(f"Generate image error: {str(e)}") 
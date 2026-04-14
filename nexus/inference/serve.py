"""Simple serving interface for NEXUS-Ω inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from nexus.config import NexusOmegaConfig
from nexus.inference.engine import InferenceEngine


@dataclass
class GenerationRequest:
    """A generation request."""

    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    task: str = "lm"


@dataclass
class GenerationResponse:
    """A generation response."""

    text: str
    tokens_generated: int
    prompt_tokens: int


class NexusServer:
    """
    Simple inference server for NEXUS-Ω.
    Manages model loading, tokenization, and generation.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[NexusOmegaConfig] = None,
        device: str = "auto",
    ):
        self.config = config or NexusOmegaConfig()
        self.model: Optional[torch.nn.Module] = None
        self.tokenizer: Optional[object] = None
        self.engine: Optional[InferenceEngine] = None

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """Load model from checkpoint."""
        from nexus.model.nexus_model import NexusOmegaModel

        self.model = NexusOmegaModel(self.config)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self.engine = InferenceEngine(self.model, self.config)

    def load_tokenizer(self, tokenizer_path: str) -> None:
        """Load tokenizer from file."""
        from nexus.tokenizer.codelingual import CodeLingualTokenizer

        self.tokenizer = CodeLingualTokenizer.load(tokenizer_path)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text from a request.

        Args:
            request: GenerationRequest with prompt and parameters.

        Returns:
            GenerationResponse with generated text.
        """
        if self.model is None or self.engine is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer() first.")

        # Tokenize
        input_ids = torch.tensor(
            [self.tokenizer.encode(request.prompt)],
            dtype=torch.long, device=self.device,
        )
        prompt_len = input_ids.shape[1]

        # Generate
        output_ids = self.engine.generate(
            input_ids,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )

        # Decode
        generated_ids = output_ids[0, prompt_len:].tolist()
        text = self.tokenizer.decode(generated_ids)

        return GenerationResponse(
            text=text,
            tokens_generated=len(generated_ids),
            prompt_tokens=prompt_len,
        )

    def batch_generate(
        self, requests: List[GenerationRequest],
    ) -> List[GenerationResponse]:
        """Generate for multiple requests."""
        return [self.generate(req) for req in requests]


import sys
import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np

# Mock folder_paths and comfy before importing nodes
sys.modules['folder_paths'] = MagicMock()
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.utils'] = MagicMock()
sys.modules['comfy.model_management'] = MagicMock()

import nodes

class TestQwenTTSImprovements(unittest.TestCase):

    def setUp(self):
        # Reset the global model cache before each test
        nodes._MODEL_CACHE = {}

    @patch('nodes.check_and_download_tokenizer')
    @patch('nodes.get_attention_implementation')
    @patch('nodes.Qwen3TTSModel')
    def test_hot_swapping_lru_cache(self, mock_model_class, mock_attn, mock_tok):
        # Setup mock model
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_attn.return_value = "sdpa"

        # Load first model
        nodes.load_qwen_model("Base", "1.7B", "cpu", "fp32")
        self.assertEqual(len(nodes._MODEL_CACHE), 1)
        key1 = list(nodes._MODEL_CACHE.keys())[0]

        # Load second model
        nodes.load_qwen_model("CustomVoice", "1.7B", "cpu", "fp32")
        self.assertEqual(len(nodes._MODEL_CACHE), 2)
        key2 = list(nodes._MODEL_CACHE.keys())[1]

        # Load third model - should trigger LRU and remove key1
        nodes.load_qwen_model("VoiceDesign", "1.7B", "cpu", "fp32")
        self.assertEqual(len(nodes._MODEL_CACHE), 2)
        self.assertNotIn(key1, nodes._MODEL_CACHE)
        self.assertIn(key2, nodes._MODEL_CACHE)

    @patch('nodes.load_qwen_model')
    @patch('nodes.split_text_by_pauses')
    def test_voice_design_variants(self, mock_split, mock_load):
        # Setup mock model and generation
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        # Mocking model.model.tts_model_type
        mock_model.model.tts_model_type = "voice_design"

        # Mocking generation output: (wavs, sr)
        # wavs should be a list of numpy arrays
        mock_wav = np.zeros((24000,))
        mock_model.generate_voice_design.return_value = ([mock_wav], 24000)

        mock_split.return_value = [("Hello", 0.0)]

        node = nodes.VoiceDesignNode()

        # Test with 3 variants
        result = node.generate(text="Hello", instruct="Gentle voice", model_choice="1.7B",
                               device="cpu", precision="fp32", language="English", num_variants=3)

        audio_data = result[0]
        waveform = audio_data['waveform']

        # Expected shape [3, 1, samples] for 3 variants
        self.assertEqual(waveform.shape[0], 3)
        self.assertEqual(mock_model.generate_voice_design.call_count, 3)

    @patch('nodes.load_qwen_model')
    @patch('nodes.split_text_by_pauses')
    def test_voice_design_06b_fallback(self, mock_split, mock_load):
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        # Set model type to custom_voice to trigger fallback logic in VoiceDesignNode
        mock_model.model.tts_model_type = "custom_voice"
        mock_model.generate_custom_voice.return_value = ([np.zeros((1000,))], 24000)
        mock_split.return_value = [("Hello", 0.0)]

        node = nodes.VoiceDesignNode()
        node.generate(text="Hello", instruct="Gentle voice", model_choice="0.6B",
                      device="cpu", precision="fp32", language="English", num_variants=1)

        # Should have called generate_custom_voice instead of generate_voice_design
        mock_model.generate_custom_voice.assert_called()
        self.assertEqual(mock_model.generate_voice_design.call_count, 0)

if __name__ == '__main__':
    unittest.main()

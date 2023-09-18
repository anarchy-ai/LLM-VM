import pytest
from unittest.mock import patch, Mock
# Assuming the ChatGPT class is in a file named onsite_llm.py
from src.llm_vm.onsite_llm import ChatGPT

# Mock the openai API interactions
class MockOpenaiAPI:
    class File:
        @staticmethod
        def create(*args, **kwargs):
            return {'id': 'mocked_file_id', 'status': 'processed'}

        @staticmethod
        def retrieve(*args, **kwargs):
            return {'status': 'processed'}

    class FineTuningJob:
        @staticmethod
        def create(*args, **kwargs):
            return {'id': 'mocked_job_id', 'status': 'succeeded'}

        @staticmethod
        def retrieve(*args, **kwargs):
            return {'status': 'succeeded', 'fine_tuned_model': 'mocked_model_id'}

    class Model:
        @staticmethod
        def delete(*args, **kwargs):
            pass

# Mock the optimizer object
class MockOptimizer:
    class storage:
        @staticmethod
        def get_model(c_id):
            return 'old_model_id'

        @staticmethod
        def set_model(c_id, new_model_id):
            pass

        @staticmethod
        def set_training_in_progress(c_id, status):
            pass

def test_fine_tune(monkeypatch):
    # Mock the openai API interactions
    monkeypatch.setattr('openai.File', MockOpenaiAPI.File)
    monkeypatch.setattr('openai.FineTuningJob', MockOpenaiAPI.FineTuningJob)
    monkeypatch.setattr('openai.Model', MockOpenaiAPI.Model)

    # Create an instance of the ChatGPT class
    chat_gpt = ChatGPT()

    # Mock the create_conversational_jsonl_file function
    mock_temp_file = Mock()
    mock_temp_file.close = Mock()
    monkeypatch.setattr('onsite_llm.create_conversational_jsonl_file', lambda x: mock_temp_file)

    # Call the fine_tune method
    dataset = [("Hello", "Hi"), ("How are you?", "I'm good")]
    optimizer = MockOptimizer()
    c_id = "test_id"
    start_function = chat_gpt.finetune(dataset, optimizer, c_id)
    start_function()

    # Assertions to check if the mocked methods were called
    mock_temp_file.close.assert_called_once()
    # Add more assertions based on expected behavior

if __name__ == "__main__":
    pytest.main()

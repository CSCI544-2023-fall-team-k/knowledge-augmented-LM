import openai
import logging

DEFAULT_ROLE = "user"

class LanguageModel:

    def __init__(self, api_key: str, model: str = None) -> None:
        self.api_key: str = api_key
        self.model: str = model
        openai.api_key = api_key

    def _generate_request_body(self, prompt: str) -> dict:
        request_body = {
            "model" : self.model,
            "messages" : [
                {
                    "role" : DEFAULT_ROLE, 
                    "content" : prompt
                }
            ]
        }
        return request_body

    def ask(self, prompt: str) -> str:
        request = self._generate_request_body(prompt)
        response = openai.ChatCompletion.create(**request)
        logging.debug(response)

        choices = response['choices']
        if len(choices) != 1:
            logging.error(f"Invalid response length: {choices}")

        return choices[0]['message']['content']
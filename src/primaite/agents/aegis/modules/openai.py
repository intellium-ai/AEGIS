from openai import OpenAI
from typing import TypeVar, Type, Any, Dict
from pydantic import BaseModel
import json

T = TypeVar("T", bound=BaseModel)


class OpenAIClient:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, max_tokens=max_new_tokens
        )

        return response.choices[0].message.content

    def generate_model(
        self, prompt: str, grammar: Type[T], max_new_tokens: int = 2048, model: str = "gpt-3.5-turbo"
    ) -> Type[T]:

        messages = [{"role": "user", "content": prompt}]
        model_schema = self._create_openai_schema(grammar)
        function_call = {"name": grammar.__name__}
        functions = [model_schema]

        # Get response from the api
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            stream=False,
            functions=functions,
            function_call=function_call,
            max_tokens=max_new_tokens,
        )
        return self._construct_model(grammar, model_schema, response)

    def remove_key_from_dict(self, d: dict[str, Any], remove_key: str) -> None:
        """
        Remove a key from a dictionary recursively

        Args:
            d (Dict[str, Any]): The dictionary
            remove_key (str): The key to remove
        """

        if isinstance(d, dict):
            for key in list(d.keys()):
                if key == remove_key:
                    del d[key]
                else:
                    self.remove_key_from_dict(d[key], remove_key)

    # These are some util functions I stole from a library called openai_function_call that does the json related grammar stuff.
    def _create_openai_schema(self, model: Type[T]) -> Dict[str, Any]:
        """
        Return the schema in the format of OpenAI's schema as jsonschema

        Returns:
            model_json_schema (dict): A dictionary in the format of OpenAI's schema as jsonschema
        """

        schema = model.model_json_schema()
        parameters = {k: v for k, v in schema.items() if k not in ("title", "description")}
        parameters["required"] = sorted(k for k, v in parameters["properties"].items() if "default" not in v)

        if "description" not in schema:
            schema["description"] = (
                f"Correctly extracted `{model.__name__}` with all the required parameters with correct types"  # type: ignore
            )

        self.remove_key_from_dict(parameters, "additionalProperties")
        self.remove_key_from_dict(parameters, "title")
        return {
            "name": schema["title"],
            "description": schema["description"],
            "parameters": parameters,
        }

    def _construct_model(
        self,
        model: Type[T],
        schema: Dict[str, Any],
        completion: Any,
        throw_error=True,
    ):
        """Execute the function from the response of an openai chat completion

        Parameters:
            model (Type[T]): The model to construct using the response.
            schema (Dict[str, Any]): The openai compatible schema to use.
            completion (Any): The response from an openai chat completion.
            throw_error (bool): Whether to throw an error if the function call is not detected. Defaults to True.

        Returns:
            model (OpenAISchema): An instance of the class
        """
        message = completion.choices[0].message

        if throw_error:
            if not hasattr(message, "function_call") or message.function_call.name != schema["name"]:
                raise Exception("No function call detected")

        function_call = message.function_call
        try:
            arguments = json.loads(function_call.arguments, strict=False)
        except json.JSONDecodeError:
            raise Exception("Unable to parse LLM output as the LLM likely generated incomplete JSON")
        return model(**arguments)

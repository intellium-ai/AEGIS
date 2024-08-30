import json
from typing import Any, Dict, Literal, Optional, Type, TypeVar
from openai import OpenAI
from collections import defaultdict
from dataclasses import dataclass

from pydantic import BaseModel
from text_generation.types import Grammar
T = TypeVar("T", bound=BaseModel)

@dataclass
class LLMMessage:
    role: Literal["user", "assistant"]
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

class OpenAIClient():

    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        """Client to use OpenAI models

        Args:
            api_key (str): The api key to use.
            model (str): The openai model to use.
        """

        self.model = model
        self.client = OpenAI(api_key=api_key)


    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
    ) -> str:

        messages = [LLMMessage("user", prompt)]

        # Convert model to openai friendly format
        model_schema = self._create_openai_schema(model)
        function_call = {"name": model.__name__}
        functions = [model_schema]


        # Get response from the api
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            functions=functions,
            function_call=function_call,
            max_tokens=1024,
            frequency_penalty=1.0,
            seed=1,
            temperature=0.0,
        )

        # Process response
        if model and model_schema:
            return self._construct_model(model, model_schema, response)
        else:
            response_str = str(response.choices[0].message.content)
            return response_str


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
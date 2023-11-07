import json
import os.path as osp
from typing import Union

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose

        if not template_name:
            template_name = "KoRAE_template"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        system_msg: Union[None, str] = None,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional system message and input
        # if a label is provided, it's also appended
        if system_msg:
            if input:
                res = self.template['prompt_input'].format(
                    system_msg=system_msg, instruction=instruction, input=input
                )
            else:
                res = self.template['prompt_no_input'].format(
                    system_msg=system_msg, instruction=instruction
                )
        else:
            if input:
                res = self.template['no_prompt_input'].format(
                    instruction=instruction, input=input
                )
            else:
                res = self.template['no_prompt_no_input'].format(
                    instruction=instruction
                )

        return res
    
    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
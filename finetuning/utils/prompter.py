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
        output: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional system message and input
        # if a label is provided, it's also appended
        if system_msg:
            res = self.template['prompt'].format(
                prompt=system_msg, 
                instruction=instruction + " " + input
            )
        else:
            res = self.template['no_prompt'].format(
                instruction=instruction + " " + input,
            )

        if output:
            res = f"{res}{output}"

        return res
    
    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
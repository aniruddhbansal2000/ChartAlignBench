import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
# from lmdeploy.vl.constants import IMAGE_TOKEN                                 {IMAGE_TOKEN} = "<IMAGE_TOKEN>"

import re

class Phi3_multi:
    instance = None
    model = None
    processor = None

    def __init__(self, device="cuda:0", temperature = -1.0):
        model_id = "microsoft/Phi-3.5-vision-instruct"  
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map= device, trust_remote_code=True, torch_dtype=torch.float16, _attn_implementation='flash_attention_2')
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.temperature = temperature
            
    def infer(self, prompt_text_template_list, images_list_list=None):
                
        prompt_list_len = len(prompt_text_template_list)

        # flatten images_list_list (can't do using func)
        images_list_flattened = []
        messages = []
        
        image_idx = 0
        for idx in range(prompt_list_len):
            
            role = "user" if idx%2==0 else "assistant"
            content = prompt_text_template_list[idx]

            # Assuming at max 1 image in the prompt
            if "image_tag" in content:
                content = content.replace("image_tag", f"<|image_{image_idx+1}|>")
                image_idx = image_idx + 1

                images_list_flattened = images_list_flattened + images_list_list[idx]

            messages.append({"role": role, "content": content})

        print(messages)
        print(images_list_flattened)
            
        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=images_list_flattened if images_list_flattened else None, return_tensors="pt").to(self.model.device)

        generation_args = {
            "max_new_tokens": 500,
        }
        if self.temperature > 0:
            generation_args["do_sample"] = True
            generation_args["temperature"] = self.temperature
        else:
            generation_args["do_sample"] = False

        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args)
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]  # Remove input tokens
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return response

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.instance, cls):
            cls.instance = object.__new__(cls)
        return cls.instance

class LlaVa_multi:
    instance = None
    model = None
    processor = None

    def __init__(self, device = "cuda:0", temperature = -1.0):
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        self.model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
        self.model.to(device)
        self.temperature = temperature

    def infer(self, prompt_text_template_list, images_list_list=None):

        prompt_list_len = len(prompt_text_template_list)

        # flatten images_list_list (can't do using func)
        images_list_flattened = []
        conversation = []
        
        image_idx = 0
        for idx in range(prompt_list_len):
            
            role = "user" if idx%2==0 else "assistant"
            content = prompt_text_template_list[idx]

            # Assuming at max 1 image in the prompt
            if "image_tag" in content:
                content = content.replace("image_tag", "")
                image_idx = image_idx + 1
                images_list_flattened = images_list_flattened + images_list_list[idx]

                conversation.append(
                    {
                    "role": role,
                    "content": [
                        {"type": "text", "text": content},
                        {"type": "image"},
                        ],
                    }
                )
                
            else:
                conversation.append(
                    {
                    "role": role,
                    "content": [
                        {"type": "text", "text": content},
                        ],
                    }
                )

        # print(prompt_text_template_list)
        print(f"prompt is: {conversation}")
        print(images_list_flattened)
        # breakpoint()
    
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = None
        if images_list_flattened:
            inputs = self.processor(images=images_list_flattened, text=prompt, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
        
        # autoregressively complete prompt
        output = None
        if self.temperature > 0:
            output = self.model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature = self.temperature)
        else: 
            output = self.model.generate(**inputs, max_new_tokens=500, do_sample=False)

        response = self.processor.decode(output[0], skip_special_tokens=True)
        
        # remove input tokens ([INST] <input text> [/INST] ``` <output text> ```)
            # incontext learning, many tokens, need final response
        input_text_end_idx = response.rfind("[/INST]") + len("[/INST]")
        response = response[input_text_end_idx:].replace("```", "")
        print(f"model response is: {response}")

        return response

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.instance, cls):
            cls.instance = object.__new__(cls)
        return cls.instance

# Run with A6000 (memory issues in A5000), as pipeline (instead of model)
class InternVL2_multi:  # version 2.5
    instance = None
    pipe = None

    def __init__(self, device = "cuda:0", temperature = 0.0):
        model_id = 'OpenGVLab/InternVL2_5-8B'
        # integer value, -1 for CPU | >=0 for corresponding GPU no
        device_int = int(device.replace("cuda:", "")) if "cuda" in device else int(-1)
        self.pipe = pipeline(model_id, backend_config=TurbomindEngineConfig(session_len=8192), device = device_int)
        self.temperature = temperature

    def infer(self, prompt_text_template_list, images_list_list=None):

        if (len(prompt_text_template_list) > 1):
            raise Exception("In-context learning not supported.")

        prompt_text = prompt_text_template_list[0]
        images_list_flattened = images_list_list[0] if images_list_list else []

        # only single image case
        if images_list_flattened:
            prompt_text = prompt_text.replace("image_tag", "<IMAGE_TOKEN>")

        gen_config = GenerationConfig(temperature = self.temperature)    # want greedy decoding (do_sample arg not there, checked 'help')

        response = None
        if images_list_flattened:
            response = self.pipe((prompt_text, images_list_flattened), gen_config=gen_config)
        else:
            response = self.pipe(prompt_text, gen_config=gen_config)
        

        return response.text

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.instance, cls):
            cls.instance = object.__new__(cls)
        return cls.instance


###----------------------------------------------------------------------------------

class Phi3:
    instance = None
    model = None
    processor = None

    def __init__(self, device="cuda:0"):
        model_id = "microsoft/Phi-3.5-vision-instruct"  
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map= device, trust_remote_code=True, torch_dtype=torch.float16, _attn_implementation='flash_attention_2')
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
    def infer(self, prompt_text_template, images_list=None):
        prompt_text = prompt_text_template
        if images_list:
            for idx in range(len(images_list)):
                tag_to_search = f"image_{idx+1}_tag"
                text_to_replace = f"<|image_{idx+1}|>"
                prompt_text = prompt_text.replace(tag_to_search, text_to_replace)

        messages = [{"role": "user", "content": prompt_text}]
        
        print(messages)
        print(images_list)

        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=prompt, images=images_list if images_list else None, return_tensors="pt").to(self.model.device)

        generation_args = {
            "max_new_tokens": 500,
            # "temperature": 0.0,
            "do_sample": False,
        }

        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args)
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]  # Remove input tokens
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return response

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.instance, cls):
            cls.instance = object.__new__(cls)
        return cls.instance

class LlaVa:
    instance = None
    model = None
    processor = None

    def __init__(self, device = "cuda:0"):
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        self.model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
        self.model.to(device)

    def infer(self, prompt_text_template, images_list=None):
        
        # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
        # Each value in "content" has to be a list of dicts with types ("text", "image") 
        prompt_text = prompt_text_template
        if images_list:
            for idx in range(len(images_list)):
                tag_to_search = f"image_{idx+1}_tag"
                prompt_text = prompt_text.replace(tag_to_search, "")
        
        conversation = None
        if not images_list:
            conversation = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    ],
                },
            ]
        elif len(images_list) == 1:
            conversation = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"},
                    ],
                },
            ]
        else:
            raise Exception("Currently not coded for multi-image processing")

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = None
        if images_list:
            inputs = self.processor(images=images_list, text=prompt, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
        
        # autoregressively complete prompt
        output = self.model.generate(**inputs, max_new_tokens=500, do_sample=False)

        response = self.processor.decode(output[0], skip_special_tokens=True)
        
        # remove input tokens ([INST] <input text> [/INST] ``` <output text> ```)
        input_text_end_idx = response.find("[/INST]") + len("[/INST]")
        response = response[input_text_end_idx:].replace("```", "")
        print(f"model response is: {response}")

        return response


    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.instance, cls):
            cls.instance = object.__new__(cls)
        return cls.instance

# Run with A6000 (memory issues in A5000), as pipeline (instead of model)
class InternVL2:    # version 2
    instance = None
    pipe = None

    def __init__(self, device = "cuda:0"):
        model_id = 'OpenGVLab/InternVL2-8B'
        # integer value, -1 for CPU | >=0 for corresponding GPU no
        device_int = int(device.replace("cuda:", "")) if "cuda" in device else int(-1)
        self.pipe = pipeline(model_id, backend_config=TurbomindEngineConfig(session_len=8192), device = device_int)

    def infer(self, prompt_text_template, images_list=None):
        prompt_text = prompt_text_template
        if images_list:
            for idx in range(len(images_list)):
                tag_to_search = f"image_{idx+1}_tag"
                prompt_text = prompt_text.replace(tag_to_search, "<IMAGE_TOKEN>")

        gen_config = GenerationConfig(temperature = 0.0)    # want greedy decoding (do_sample arg not there, checked 'help')

        response = None
        if images_list:
            response = self.pipe((prompt_text, images_list), gen_config=gen_config)
        else:
            response = self.pipe(prompt_text, gen_config=gen_config)
        
        return response.text

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.instance, cls):
            cls.instance = object.__new__(cls)
        return cls.instance

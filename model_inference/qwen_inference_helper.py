import torch
from transformers import AutoProcessor

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from qwen_vl_utils import process_vision_info


class Qwen_multi:
    instance = None
    model = None
    processor = None

    def __init__(self, device = "cuda:0", temperature = -1.0):
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map=device)
        self.processor = AutoProcessor.from_pretrained(model_id, revision="refs/pr/24") # image processor config issue in JSON (branch not merged in main yet)
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
                content = content.replace("image_tag", "")
                image_idx = image_idx + 1
                images_list_flattened = images_list_flattened + images_list_list[idx]

                messages.append(
                    {
                    "role": role,
                    "content": [
                            {
                                "type": "image",
                                "image": images_list_list[idx][0],
                            },
                            {"type": "text", "text": content},
                        ],
                    }
                )
                
            else:
                messages.append(
                    {
                    "role": role,
                    "content": [
                        {"type": "text", "text": content},
                        ],
                    }
                )

        print(f"prompt is: {messages}")
        print(f"image list: {images_list_flattened}")
        # breakpoint()

        # Preparation for inference
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = None
        if images_list_flattened:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
            )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = None
        if self.temperature > 0:
            generated_ids = self.model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature = self.temperature)
        else:
            generated_ids = self.model.generate(**inputs, max_new_tokens=500, do_sample=False)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        output_text = output_text[0].replace("```", "")
        # print(f"response is: {output_text}")
        

        return output_text

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.instance, cls):
            cls.instance = object.__new__(cls)
        return cls.instance


class Qwen:
    instance = None
    model = None
    processor = None

    def __init__(self, device = "cuda:0"):
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map=device)
        self.processor = AutoProcessor.from_pretrained(model_id, revision="refs/pr/24") # image processor config issue in JSON (branch not merged in main yet)

    def infer(self, prompt_text_template, images_list=None):
        
        prompt_text = prompt_text_template
        if images_list:
            for idx in range(len(images_list)):
                tag_to_search = f"image_{idx+1}_tag"
                prompt_text = prompt_text.replace(tag_to_search, "")
        
        messages = None
        if not images_list:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
        elif len(images_list) == 1:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": images_list[0],
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
        else:
            raise Exception("Currently not coded for multi-image processing")

        # Preparation for inference
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = None
        if images_list:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
            )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=500, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        output_text = output_text[0].replace("```", "")
        # print(f"response is: {output_text}")
        

        return output_text

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.instance, cls):
            cls.instance = object.__new__(cls)
        return cls.instance

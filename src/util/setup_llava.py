from transformers import AutoProcessor, LlavaForConditionalGeneration


def load_llava_model_and_processor():
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    # not enough memory to run on GPU
    return model, processor

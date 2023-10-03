import torch
from compel import Compel, ReturnedEmbeddingsType

# TODO: refactor class
class SDXLCompelHelper:
    def __init__(self,
                 tokenizer,
                 text_encoder,
                 tokenizer_2,
                 text_encoder_2):
        self.base_compel_1 = Compel(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=False,
        )
        self.base_compel_2 = Compel(
            tokenizer=tokenizer_2,
            text_encoder=text_encoder_2,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=True,
        )

    def get_embeddings(self, prompt, prompt_2, negative_prompt, negative_prompt_2):
        # set embeds
        base_positive_prompt_embeds_1 = self.base_compel_1(prompt)
        base_positive_prompt_embeds_2, base_positive_prompt_pooled = self.base_compel_2(prompt_2)
        base_negative_prompt_embeds_1 = self.base_compel_1(negative_prompt)
        base_negative_prompt_embeds_2, base_negative_prompt_pooled = self.base_compel_2(negative_prompt_2)

        # Pad the conditioning tensors to ensure thet they all have the same length
        (base_positive_prompt_embeds_2, base_negative_prompt_embeds_2) = self.base_compel_2.pad_conditioning_tensors_to_same_length([base_positive_prompt_embeds_2, base_negative_prompt_embeds_2])

        # Concatenate the cconditioning tensors corresponding to both the set of prompts
        base_positive_prompt_embeds = torch.cat((base_positive_prompt_embeds_1, base_positive_prompt_embeds_2), dim=-1)
        base_negative_prompt_embeds = torch.cat((base_negative_prompt_embeds_1, base_negative_prompt_embeds_2), dim=-1)

        return (base_positive_prompt_embeds, base_positive_prompt_pooled, base_negative_prompt_embeds, base_negative_prompt_pooled)




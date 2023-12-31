import torch
from compel import Compel, ReturnedEmbeddingsType

# TODO: refactor class
class SDXLCompelHelper:
    def __init__(self,
                 tokenizer,
                 text_encoder,
                 tokenizer_2,
                 text_encoder_2):
        self.__tokenizer = tokenizer
        self.__text_encoder = text_encoder
        self.__tokenizer_2 = tokenizer_2
        self.__text_encoder_2 = text_encoder_2
        
    #TODO: refactor method
    def get_embeddings(self, prompt, prompt_2, negative_prompt, negative_prompt_2):
        if self.__tokenizer == None and self.__text_encoder == None:
            # init compel
            base_compel = Compel(
            tokenizer=self.__tokenizer_2,
            text_encoder=self.__text_encoder_2,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=True,
            )

            base_positive_prompt_embeds, base_positive_prompt_pooled = base_compel(prompt+" "+prompt_2)
            base_negative_prompt_embeds, base_negative_prompt_pooled = base_compel(negative_prompt+" "+negative_prompt_2)
            base_positive_prompt_embeds, base_negative_prompt_embeds = base_compel.pad_conditioning_tensors_to_same_length([
                    base_positive_prompt_embeds, base_negative_prompt_embeds])
        else:
            if prompt_2 == "" and negative_prompt_2 == "":
                # init compel
                base_compel = Compel(
                    tokenizer=[self.__tokenizer, self.__tokenizer_2],
                    text_encoder=[self.__text_encoder, self.__text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True]
                )

                base_positive_prompt_embeds, base_positive_prompt_pooled = base_compel(prompt)
                base_negative_prompt_embeds, base_negative_prompt_pooled = base_compel(negative_prompt)
                base_positive_prompt_embeds, base_negative_prompt_embeds = base_compel.pad_conditioning_tensors_to_same_length([
                    base_positive_prompt_embeds, base_negative_prompt_embeds])
            else:
                # init compels
                base_compel_1 = Compel(
                    tokenizer=self.__tokenizer,
                    text_encoder=self.__text_encoder,
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=False,
                )
                base_compel_2 = Compel(
                    tokenizer=self.__tokenizer_2,
                    text_encoder=self.__text_encoder_2,
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=True,
                )
                # set embeds
                base_positive_prompt_embeds_1 = base_compel_1(prompt)
                base_positive_prompt_embeds_2, base_positive_prompt_pooled = base_compel_2(prompt_2)
                base_negative_prompt_embeds_1 = base_compel_1(negative_prompt)
                base_negative_prompt_embeds_2, base_negative_prompt_pooled = base_compel_2(negative_prompt_2)

                # Pad the conditioning tensors to ensure thet they all have the same length
                (base_positive_prompt_embeds_2, base_negative_prompt_embeds_2) = base_compel_2.pad_conditioning_tensors_to_same_length([base_positive_prompt_embeds_2, base_negative_prompt_embeds_2])

                # Concatenate the conditioning tensors corresponding to both the set of prompts
                base_positive_prompt_embeds = torch.cat((base_positive_prompt_embeds_1, base_positive_prompt_embeds_2), dim=-1)
                base_negative_prompt_embeds = torch.cat((base_negative_prompt_embeds_1, base_negative_prompt_embeds_2), dim=-1)
        
        return (base_positive_prompt_embeds, base_positive_prompt_pooled, base_negative_prompt_embeds, base_negative_prompt_pooled)
    
    def get_embeds_args(self, prompt, prompt_2, negative_prompt, negative_prompt_2) -> dict[str, torch.Tensor]:
        (base_positive_prompt_embeds, base_positive_prompt_pooled, base_negative_prompt_embeds, base_negative_prompt_pooled) = self.get_embeddings(prompt, prompt_2, negative_prompt, negative_prompt_2)
        # set embeds dict
        embeds_args = {}
        embeds_args['prompt_embeds'] = base_positive_prompt_embeds
        embeds_args['pooled_prompt_embeds'] = base_positive_prompt_pooled
        embeds_args['negative_prompt_embeds'] = base_negative_prompt_embeds
        embeds_args['negative_pooled_prompt_embeds'] = base_negative_prompt_pooled
        return embeds_args